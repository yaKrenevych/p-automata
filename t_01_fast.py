import random
from typing import Dict, List, Tuple, Iterable, Optional
import itertools


# =============================
#  Модульні мапи ("захардкожені" для p)
# =============================
class SigmaModP:
    """
    Попередньо обчислює мапи x -> (x-1) mod p та x -> (x+1) mod p для символів '0'..'p-1'.
    Повертає результат як рядки без перетворення типів під час роботи автомата.
    """
    def __init__(self, p: int):
        if p < 2:
            raise ValueError("p має бути >= 2")
        self.p = p
        self.minus1_map: Dict[str, str] = {}
        self.plus1_map: Dict[str, str] = {}
        for i in range(p):
            s = str(i)
            self.minus1_map[s] = str((i - 1) % p)
            self.plus1_map[s] = str((i + 1) % p)

    def minus1(self, x: str) -> str:
        return self.minus1_map[x]

    def plus1(self, x: str) -> str:
        return self.plus1_map[x]


# =============================
#  Узагальнений автомат Мілі
# =============================
class MealyAutomaton:
    def __init__(self, name: str, p: int,
                 transition_table: Dict[str, List[str]],
                 inv_transition_table: Dict[str, List[str]],
                 sigma: SigmaModP):
        self.name = name  # 'a' або 'b'
        self.p = p
        self.tr = transition_table
        self.tr_inv = inv_transition_table
        self.sigma = sigma
        # Розрахунок кількості станів за довжиною рядків у таблиці
        any_row = next(iter(transition_table.values()))
        self.num_states = len(any_row)

    # --- Вихідні функції λ ---
    # За умовою, тільки стани №5 та №10 змінюють символ на (x-1) mod p (або обернено (x+1) mod p)
    def _state_index(self, state: str) -> int:
        # state формат: 'a1', 'a2', ..., 'b1', 'b2', ...
        return int(state[1:])  # 1..N

    def lam(self, x: str, state: str) -> str:
        i = self._state_index(state)
        if i in (5, 10):
            return self.sigma.minus1(x)
        return x

    def lam_inv(self, x: str, state: str) -> str:
        i = self._state_index(state)
        if i in (5, 10):
            return self.sigma.plus1(x)
        return x

    # --- Перехідні функції ψ ---
    @staticmethod
    def _col_index(state: str) -> int:
        return int(state[1:]) - 1  # 0..N-1

    @staticmethod
    def _select_row(table: Dict[str, List[str]], x: str) -> List[str]:
        # Якщо для символа є окремий рядок — беремо його, інакше узагальнений 'x'
        if x in table:
            return table[x]
        return table['x']

    def psi(self, state: str, x: str) -> str:
        row = self._select_row(self.tr, x)
        j = self._col_index(state)
        return row[j]

    def psi_inv(self, state: str, x: str) -> str:
        row = self._select_row(self.tr_inv, x)
        j = self._col_index(state)
        return row[j]

    # --- Застосування до слова ---
    def apply(self, init_state: str, word: str) -> str:
        state = init_state
        out_chars: List[str] = []
        for ch in word:
            out_ch = self.lam(ch, state)
            out_chars.append(out_ch)
            state = self.psi(state, ch)
        return ''.join(out_chars)

    def apply_inv(self, init_state: str, word: str) -> str:
        state = init_state
        out_chars: List[str] = []
        for ch in word:
            out_ch = self.lam_inv(ch, state)
            out_chars.append(out_ch)
            state = self.psi_inv(state, ch)
        return ''.join(out_chars)


# =============================
#  Конкретні автомати A та B
# =============================
class AutomatonA(MealyAutomaton):
    @staticmethod
    def build_transition_table_A() -> Dict[str, List[str]]:
        return {
            '0': ['a2', 'a4', 'a1', 'a5', 'a4', 'a8', 'a1', 'a9', 'a4', 'a1', 'a1', 'a13', 'a12', 'a1'],
            '1': ['a3', 'a12', 'a1', 'a7', 'a12', 'a8', 'a1', 'a10', 'a12', 'a1', 'a1', 'a13', 'a12', 'a1'],
            'x': ['a3', 'a12', 'a1', 'a6', 'a12', 'a8', 'a1', 'a11', 'a12', 'a1', 'a1', 'a14', 'a12', 'a1']
        }

    @staticmethod
    def build_inv_transition_table_A(p: int) -> Dict[str, List[str]]:
        return {
            str(p - 1): ['a3', 'a12', 'a1', 'a6', 'a4', 'a8', 'a1', 'a11', 'a12', 'a1', 'a1', 'a14', 'a12', 'a1'],
            '0':        ['a2', 'a4', 'a1', 'a5', 'a12', 'a8', 'a1', 'a9',  'a4',  'a1', 'a1', 'a13', 'a12', 'a1'],
            '1':        ['a3', 'a12', 'a1', 'a7', 'a12', 'a8', 'a1', 'a10', 'a12', 'a1', 'a1', 'a13', 'a12', 'a1'],
            'x':        ['a3', 'a12', 'a1', 'a6', 'a12', 'a8', 'a1', 'a11', 'a12', 'a1', 'a1', 'a14', 'a12', 'a1']
        }

    def __init__(self, p: int, sigma: SigmaModP):
        super().__init__(
            name='a', p=p,
            transition_table=self.build_transition_table_A(),
            inv_transition_table=self.build_inv_transition_table_A(p),
            sigma=sigma,
        )


class AutomatonB(MealyAutomaton):
    @staticmethod
    def build_transition_table_B() -> Dict[str, List[str]]:
        return {
            '0': ['b3', 'b4', 'b1', 'b7', 'b4', 'b8', 'b1', 'b10', 'b4', 'b1', 'b1', 'b13', 'b12', 'b1'],
            '1': ['b2', 'b12', 'b1', 'b5', 'b12', 'b8', 'b1', 'b9',  'b12', 'b1', 'b1', 'b13', 'b12', 'b1'],
            'x': ['b3', 'b12', 'b1', 'b6', 'b12', 'b8', 'b1', 'b11', 'b12', 'b1', 'b1', 'b14', 'b12', 'b1']
        }

    @staticmethod
    def build_inv_transition_table_B(p: int) -> Dict[str, List[str]]:
        return {
            str(p - 1): ['b3', 'b12', 'b1', 'b6', 'b4', 'b8', 'b1', 'b11', 'b12', 'b1', 'b1', 'b14', 'b12', 'b1'],
            '0':        ['b3', 'b4',  'b1', 'b7', 'b12', 'b8', 'b1', 'b10', 'b4',  'b1', 'b1', 'b13', 'b12', 'b1'],
            '1':        ['b2', 'b12', 'b1', 'b5', 'b12', 'b8', 'b1', 'b9',  'b12', 'b1', 'b1', 'b13', 'b12', 'b1'],
            'x':        ['b3', 'b12', 'b1', 'b6', 'b12', 'b8', 'b1', 'b11', 'b12', 'b1', 'b1', 'b14', 'b12', 'b1']
        }

    def __init__(self, p: int, sigma: SigmaModP):
        super().__init__(
            name='b', p=p,
            transition_table=self.build_transition_table_B(),
            inv_transition_table=self.build_inv_transition_table_B(p),
            sigma=sigma,
        )


# =============================
#  Допоміжні функції
# =============================

def build_alphabet(p: int) -> List[str]:
    return [str(i) for i in range(p)]


def to_base_p(n: int, p: int) -> List[str]:
    if n == 0:
        return ['0']
    digits: List[str] = []
    while n > 0:
        digits.append(str(n % p))
        n //= p
    return digits[::-1]


def build_w_from_g(g: Iterable[Tuple[str, int]], p: int) -> str:
    """
    g — послідовність пар ('a'|'b', k), k != 0.
    Синтезує слово w за вашою схемою: стартуємо з '00',
    потім для кожного генератора додаємо префікс ('0' для 'a', '1' для 'b'),
    далі цифри представлення |k|-1 у базі p, потім '22', потім тригерний біт і xi.
    (Логіка з оригіналу збережена.)
    """
    w = '00'
    for gen, k in g:
        if k == 0:
            continue
        vi = to_base_p(abs(k) - 1, p)
        xi = '1' if k > 0 else str(p - 1)
        if gen == 'a':
            w += '0' + ''.join(vi) + '22' + '1' + xi
        else:
            w += '1' + ''.join(vi) + '22' + '0' + xi
    return w


def alternating_g(first_gen: str, exponents: Iterable[int]) -> List[Tuple[str, int]]:
    """Формує редуковане слово g зі стартового генератора та послідовності ненульових степенів."""
    gens = []
    cur = first_gen
    for k in exponents:
        if k == 0:
            continue
        gens.append((cur, k))
        cur = 'b' if cur == 'a' else 'a'
    return gens


def generate_random_reduced_word(length: int = 5, max_exp: int = 10) -> Tuple[str, Tuple[int, ...]]:
    if length <= 0:
        return 'a', tuple()
    first_gen = random.choice(['a', 'b'])
    exps: List[int] = []
    for _ in range(length):
        k = 0
        while k == 0:
            k = random.randint(-max_exp, max_exp)
        exps.append(k)
    return first_gen, tuple(exps)


# =============================
#  Застосування елементів групи (черги генераторів)
# =============================

def apply_generator(word: str, gen: str, k: int, p: int,
                    A: AutomatonA, B: AutomatonB) -> str:
    if gen == 'a':
        autom = A
        init_state = 'a1'
    else:
        autom = B
        init_state = 'b1'

    times = abs(k)
    if times == 0:
        return word

    if k > 0:
        for _ in range(times):
            word = autom.apply(init_state, word)
    else:
        for _ in range(times):
            word = autom.apply_inv(init_state, word)
    return word


def apply_g(g: Iterable[Tuple[str, int]], word: str, p: int,
            A: AutomatonA, B: AutomatonB) -> str:
    for gen, k in g:
        word = apply_generator(word, gen, k, p, A, B)
    return word


# =============================
# DP для підрахунку фіксованих слів
# =============================


def count_fixed_words(g: Iterable[Tuple[str, int]], n: int, p: int,
    A: AutomatonA, B: AutomatonB,
    show_words: bool = False,
    outfile: Optional[str] = None) -> int:
    """
    Повертає кількість слів довжини n, які залишаються незмінними після застосування g.
    Якщо show_words=True — друкує їх у консоль.
    Якщо outfile заданий — записує результат у файл (одне слово на рядок).
    """
    if n < 0:
        raise ValueError("n має бути невід'ємним")
    alpha = build_alphabet(p)
    fixed_words: List[str] = []
    total = 0

    for tup in itertools.product(alpha, repeat=n):
        w = ''.join(tup)
        gw = apply_g(g, w, p, A, B)
        if gw == w:
            total += 1
            fixed_words.append(w)
            if show_words:
                print(f"w={w}, g(w)={gw}")

    if outfile:
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(f"Fixed words of length {n} (count={total}):\n")
            for s in fixed_words:
                f.write(f"w={s}, g(w)={s}\n")

    return total


def count_fixed_recursive(n: int, p: int, g: List[Tuple[str, int]],
                          A: AutomatonA, B: AutomatonB,
                          show_words: bool = False,
                          outfile: Optional[str] = None) -> int:
    # Підготовка ланцюжка g
    sequence = []
    for gen, k in g:
        autom = A if gen == 'a' else B
        is_inv = k < 0
        for _ in range(abs(k)):
            sequence.append((autom, is_inv))

    initial_states = tuple(('a1' if gen == 'a' else 'b1') for gen, k in g for _ in range(abs(k)))
    fixed_words_list = []

    # Внутрішня функція обходу (Backtracking)
    # якщо треба бачити самі слова
    def collect_words(rem_n, current_states, current_path):
        if rem_n == 0:      #Базовий випадок рекурсії. Якщо ми дійшли до кінця (довжина слова сформована), ми «склеюємо» символи в рядок
            word = "".join(current_path)
            fixed_words_list.append(word)       #Додаємо знайдене нерухоме слово до загального списку
            if show_words:
                print(f"w={word}, g(w)={word}")
            return 1    #Повертаємо одиницю, щоб лічильник на вищих рівнях рекурсії знав, що знайдено одне повне слово

        count = 0       #Ініціалізуємо суму нерухомих слів для поточної
        for i in range(p):
            char = str(i)
            temp_char = char
            new_states = []     #Тут ми будемо накопичувати стани, у які перейдуть автомати після обробки цього символу
            for idx, (autom, is_inv) in enumerate(sequence):
                st = current_states[idx]
                if is_inv:
                    out_char = autom.lam_inv(temp_char, st)         #Вихід одного автомата стає входом для наступного (композиція функцій)
                    nxt_st = autom.psi_inv(st, temp_char)
                else:
                    out_char = autom.lam(temp_char, st)
                    nxt_st = autom.psi(st, temp_char)
                temp_char = out_char
                new_states.append(nxt_st)

            if temp_char == char:       #Якщо після проходження через весь ланцюжок g символ на виході збігається з входом, то ця гілка потенційно може стати нерухомим словом
                count += collect_words(rem_n - 1, tuple(new_states), current_path + [char])
        return count

    # Внутрішня функція з мемоізацією (Швидка)
    # Використовується, коли потрібна тільки кількість

    memo = {}   #Словник-кеш, що зберігається поза функцією (або в її замиканні). Він пам'ятає результати для кожної комбінації

    def fast_count(rem_n, current_states):      #rem_n - Скільки символів ще треба обробити
        state_key = (rem_n, current_states)     #Створюємо унікальний ключ для кешу
        if state_key in memo: return memo[state_key]
        if rem_n == 0: return 1

        count = 0
        for i in range(p):
            char = str(i)
            temp_char = char
            new_states = []
            for idx, (autom, is_inv) in enumerate(sequence):
                st = current_states[idx]
                out_char = autom.lam_inv(temp_char, st) if is_inv else autom.lam(temp_char, st)
                nxt_st = autom.psi_inv(st, temp_char) if is_inv else autom.psi(st, temp_char)
                temp_char, new_states = out_char, new_states + [nxt_st]

            if temp_char == char:
                count += fast_count(rem_n - 1, tuple(new_states))

        memo[state_key] = count
        return count

    # Вибір логіки
    if show_words or outfile:
        total = collect_words(n, initial_states, [])
    else:
        total = fast_count(n, initial_states)

    # Запис у файл, як у вашій версії
    if outfile:
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(f"Fixed words of length {n} (count={total}):\n")
            for s in fixed_words_list:
                f.write(f"w={s}, g(w)={s}\n")

    return total

def generate_unique_g(seen: set, length: int = 5, max_exp: int = 10) -> List[Tuple[str, int]]:
    while True:
        first, exps = generate_random_reduced_word(length=length, max_exp=max_exp)
        g = tuple(alternating_g(first, exps))  # tuple, щоб можна було класти у set
        if g not in seen:
            seen.add(g)
            return list(g)


def log_statistics(g: Iterable[Tuple[str, int]], n: int, fixed_count: int,
                   outfile: str = "statistics.txt") -> None:
    """
    Додає у файл рядок виду:
    g = [...], n=7, fixed words = 1863
    """
    with open(outfile, "a", encoding="utf-8") as f:
        f.write(f"g = {list(g)} n={n}, fixed words = {fixed_count}\n")


def cnt_average_for1(p, n, cnt):
    res = cnt / (p**n)
    return res


################

#ми шукаємо найкоротший шлях у дереві, який веде до «активного» стану, що змінює символ
#Тут нам потрібен пошук у ширину (BFS). Ми хочемо знайти найкоротше слово, яке автомат змінює.
#Оскільки ми шукаємо найменше n_2, ми перевіряємо спочатку всі слова довжини 1, потім 2, і так далі, поки не знайдемо перше слово, для якого gw = w
#нам потрібне найкоротше слово, яке приводить ланцюжок автоматів g до стану, де хоча б один символ буде змінений (у наших автоматах це стани 5 або 10).


from collections import deque

def find_minimal_non_fixed_word(g: List[Tuple[str, int]], p: int, A: AutomatonA, B: AutomatonB) -> Tuple[int, str]:
    """
    Знаходить найкоротше слово w, таке що g(w) != w.
    Використовує BFS для гарантії мінімальності довжини n2.
    """

    sequence = []
    for gen, k in g:
        autom = A if gen == 'a' else B
        for _ in range(abs(k)):
            sequence.append((autom, k < 0))

    initial_states = tuple(('a1' if gen == 'a' else 'b1') for gen, k in g for _ in range(abs(k)))

    # Черга для BFS: (поточні_стани, накопичене_слово)
    queue = deque([(initial_states, "")])
    visited_states = {initial_states}  # Щоб не ходити по колу

    while queue:
        curr_states, path = queue.popleft()

        for i in range(p):
            char = str(i)
            temp_char = char
            new_states = []

            # Проганяємо символ через ланцюжок g
            for idx, (autom, is_inv) in enumerate(sequence):
                st = curr_states[idx]
                if is_inv:
                    out_char = autom.lam_inv(temp_char, st)
                    nxt_st = autom.psi_inv(st, temp_char)
                else:
                    out_char = autom.lam(temp_char, st)
                    nxt_st = autom.psi(st, temp_char)
                temp_char = out_char
                new_states.append(nxt_st)

            new_word = path + char

            # Якщо символ змінився — ми знайшли n2!
            if temp_char != char:
                return len(new_word), new_word

            # Якщо не змінився, додаємо в чергу для перевірки наступного рівня довжини
            new_states_tuple = tuple(new_states)
            if new_states_tuple not in visited_states:
                visited_states.add(new_states_tuple)
                queue.append((new_states_tuple, new_word))

    return -1, ""  # Теоретично неможливо для нетривіального g


def collect_full_statistics(iterations: int, p: int, A: AutomatonA, B: AutomatonB, filename: str):
    seen_g = set()
    results = []

    with open(filename, "w", encoding="utf-8") as f:
        f.write("g | n1 | w1g | n2 | w2g\n")
        f.write("-" * 50 + "\n")

        for _ in range(iterations):
            g = generate_unique_g(seen_g, length=5, max_exp=5)

            # Дані з статті (w1)
            w1g = build_w_from_g(g, p)
            n1 = len(w1g)

            # Пошук мінімального n2 (BFS)
            n2, w2g = find_minimal_non_fixed_word(g, p, A, B)

            # Запис результату
            line = f"{list(g)} | {n1} | {w1g} | {n2} | {w2g}"
            print(line)
            f.write(line + "\n")

            results.append({
                'g': g, 'n1': n1, 'w1g': w1g, 'n2': n2, 'w2g': w2g
            })

    return results


if __name__ == '__main__':
    p = 3
    sigma = SigmaModP(p)
    A = AutomatonA(p, sigma)
    B = AutomatonB(p, sigma)

    seen_g = set()
    n = 50
    iterations = 2000
    suma = 0

    for _ in range(iterations):
        # Рандомно генеруємо g
        g = generate_unique_g(seen_g, length=20, max_exp=20)

        cnt = count_fixed_recursive(
            n=n,
            p=p,
            g=g,
            A=A,
            B=B,
            show_words=False,  # Якщо True - буде повільніше, бо не працює кеш
            outfile=None
        )

        suma += cnt / (p ** n)
        log_statistics(g, n, cnt, outfile="n50_len20_exp20_iter2000.txt")

    print(f'Average res for n = {n} is {suma / iterations}')
"""

if __name__ == '__main__':
    p = 3
    sigma = SigmaModP(p)
    A = AutomatonA(p, sigma)
    B = AutomatonB(p, sigma)
    output_file = "full_group_stats.txt"

    collect_full_statistics(
        iterations=2000,
        p=p,
        A=A,
        B=B,
        filename=output_file
    )

    print(f"\nСтатистику зібрано у файл: {output_file}")"""