#include <iostream>
#include <stdexcept>
#include <limits>

#include "CondFormats/Serialization/interface/Equal.h"

template <typename T>
void same(const T &first, const T &second) {
  std::cout << "Same " << typeid(T).name() << std::endl;
  if (not cond::serialization::equal(first, second))
    throw std::logic_error("Objects are not equal.");
}

template <typename T>
void diff(const T &first, const T &second) {
  std::cout << "Diff " << typeid(T).name() << std::endl;
  if (cond::serialization::equal(first, second))
    throw std::logic_error("Objects are not different.");
}

template <typename T>
void checkFloatingPoint() {
  const T zero(0);
  const T nzero(-zero);
  const T first(1);
  const T second(2);
  const T inf(std::numeric_limits<T>::infinity());
  const T ninf(-inf);
  const T qnan(std::numeric_limits<T>::quiet_NaN());
  const T nqnan(-qnan);
  const T snan(std::numeric_limits<T>::signaling_NaN());
  const T nsnan(-snan);

  auto positive = [](T x) {
    if (std::signbit(x))
      throw std::logic_error("Object is not positive.");
  };
  auto negative = [](T x) {
    if (not std::signbit(x))
      throw std::logic_error("Object is not negative.");
  };

  // Check if we got the signbit right
  positive(zero);
  negative(nzero);
  positive(first);
  positive(second);
  positive(inf);
  negative(ninf);
  positive(qnan);
  negative(nqnan);
  positive(snan);
  negative(nsnan);

  same(zero, zero);
  same(zero, nzero);

  same(first, first);
  diff(first, second);

  same(inf, inf);
  same(ninf, ninf);
  diff(inf, ninf);

  // see notes in SerializationEqual.h
  same(qnan, qnan);
  same(qnan, nqnan);
  same(qnan, snan);
  same(qnan, nsnan);
  same(nqnan, nqnan);
  same(nqnan, snan);
  same(nqnan, nsnan);
  same(snan, snan);
  same(snan, nsnan);
  same(nsnan, nsnan);

  diff(zero, first);
  diff(zero, inf);
  diff(zero, qnan);
  diff(first, inf);
  diff(first, qnan);
  diff(inf, qnan);
}

template <typename T>
void checkSequence() {
  // empty
  same(T({}), T({}));

  // diff size
  diff(T({}), T({{1}, {2}, {3}}));
  diff(T({{1}, {2}}), T({{1}, {2}, {3}}));

  // same size
  same(T({{1}, {2}, {3}}), T({{1}, {2}, {3}}));
  diff(T({{1}, {2}, {3}}), T({{4}, {2}, {3}}));
  diff(T({{1}, {2}, {3}}), T({{1}, {4}, {3}}));
  diff(T({{1}, {2}, {3}}), T({{1}, {2}, {4}}));
}

// This applies for both ordered and unordered mappings
template <typename T>
void checkCommonMapping() {
  // empty
  same(T({}), T({}));

  // same
  same(T({{1, 2}, {2, 3}}), T({{1, 2}, {2, 3}}));

  // diff size
  diff(T({}), T({{1, 2}, {2, 3}}));
  diff(T({{1, 2}}), T({{1, 2}, {2, 3}}));

  // diff keys
  diff(T({{1, 2}, {2, 3}}), T({{2, 2}, {2, 3}}));
  diff(T({{1, 2}, {2, 3}}), T({{1, 2}, {3, 3}}));

  // diff values
  diff(T({{1, 2}, {2, 3}}), T({{1, 3}, {2, 3}}));
  diff(T({{1, 2}, {2, 3}}), T({{1, 2}, {2, 4}}));
}

int main() {
  // integral
  same(false, false);
  diff(false, true);
  same('1', '1');
  diff('1', '2');
  same(static_cast<unsigned char>('1'), static_cast<unsigned char>('1'));
  diff(static_cast<unsigned char>('1'), static_cast<unsigned char>('2'));
  same(L'1', L'1');
  diff(L'1', L'2');
  same(u'1', u'1');
  diff(u'1', u'2');
  same(U'1', U'1');
  diff(U'1', U'2');
  same(static_cast<short>(1), static_cast<short>(1));
  diff(static_cast<short>(1), static_cast<short>(2));
  same(static_cast<unsigned short>(1), static_cast<unsigned short>(1));
  diff(static_cast<unsigned short>(1), static_cast<unsigned short>(2));
  same(1, 1);
  diff(1, 2);
  same(1u, 1u);
  diff(1u, 2u);
  same(1l, 1l);
  diff(1l, 2l);
  same(1ul, 1ul);
  diff(1ul, 2ul);
  same(1ll, 1ll);
  diff(1ll, 2ll);
  same(1ull, 1ull);
  diff(1ull, 2ull);

  // enum
  enum enums { enum1, enum2 };
  same(enum1, enum1);
  diff(enum1, enum2);

  // floating point
  checkFloatingPoint<float>();
  checkFloatingPoint<double>();
  checkFloatingPoint<long double>();

  // string
  same(std::string("hi"), std::string("hi"));
  diff(std::string("hi"), std::string("hj"));
  diff(std::string("hi"), std::string("hi2"));

  // bitset
  same(std::bitset<3>("101"), std::bitset<3>("101"));
  diff(std::bitset<3>("101"), std::bitset<3>("001"));
  diff(std::bitset<3>("101"), std::bitset<3>("111"));
  diff(std::bitset<3>("101"), std::bitset<3>("100"));

  // pair
  same(std::make_pair(1, '1'), std::make_pair(1, '1'));
  diff(std::make_pair(1, '1'), std::make_pair(2, '1'));
  diff(std::make_pair(1, '1'), std::make_pair(1, '2'));

  // tuple
  same(std::make_tuple(1, '1', 1.f), std::make_tuple(1, '1', 1.f));
  diff(std::make_tuple(1, '1', 1.f), std::make_tuple(2, '1', 1.f));
  diff(std::make_tuple(1, '1', 1.f), std::make_tuple(1, '2', 1.f));
  diff(std::make_tuple(1, '1', 1.f), std::make_tuple(1, '1', 2.f));

  // pointer
  int i1 = 1, i1b = 1, i2 = 2;
  int *pi1 = &i1, *pi1b = &i1b, *pi2 = &i2, *pin = nullptr;
  same(pi1, pi1);   // same addr
  same(pi1, pi1b);  // diff addr, same value
  diff(pi1, pi2);   // diff addr, diff value
  same(pin, pin);   // nullptr, same
  diff(pin, pi1);   // nullptr, diff, avoid dereferenciation, first
  diff(pi1, pin);   // nullptr, diff, avoid dereferenciation, second

  std::unique_ptr<int> ui1(pi1), u2i1(pi1), ui1b(pi1b), ui2(pi2), uin(pin);
  same(ui1, ui1);
  same(ui1, u2i1);  // diff object, same addr (even if it should not happen)
  same(ui1, ui1b);
  diff(ui1, ui2);
  same(uin, uin);
  diff(uin, ui1);
  diff(ui1, uin);
  ui1.release();
  u2i1.release();
  ui1b.release();
  ui2.release();
  uin.release();

  auto deleter = [](int *) {};
  std::shared_ptr<int> si1(pi1, deleter), s2i1(pi1, deleter), si1b(pi1b, deleter), si2(pi2, deleter), sin(pin, deleter);
  same(si1, si1);
  same(si1, s2i1);  // diff object, same addr (may happen since it is a shared_ptr)
  same(si1, si1b);
  diff(si1, si2);
  same(sin, sin);
  diff(sin, si1);
  diff(si1, sin);
  si1.reset();
  s2i1.reset();
  si1b.reset();
  si2.reset();
  sin.reset();

  std::shared_ptr<int> bsi1(pi1, deleter), bs2i1(pi1, deleter), bsi1b(pi1b, deleter), bsi2(pi2, deleter),
      bsin(pin, deleter);
  same(bsi1, bsi1);
  same(bsi1, bs2i1);  // diff object, same addr (may happen since it is a shared_ptr)
  same(bsi1, bsi1b);
  diff(bsi1, bsi2);
  same(bsin, bsin);
  diff(bsin, bsi1);
  diff(bsi1, bsin);
  bsi1.reset();
  bs2i1.reset();
  bsi1b.reset();
  bsi2.reset();
  bsin.reset();

  // C-style array
  int a123[] = {1, 2, 3}, a123b[] = {1, 2, 3}, a223[] = {2, 2, 3}, a133[] = {1, 3, 3}, a124[] = {1, 2, 4};
  same(a123, a123);
  same(a123, a123b);
  diff(a123, a223);
  diff(a123, a133);
  diff(a123, a124);
  same("hi", "hi");
  diff("hi", "hj");

  // array
  same(std::array<int, 3>{{1, 2, 3}}, std::array<int, 3>{{1, 2, 3}});
  diff(std::array<int, 3>{{1, 2, 3}}, std::array<int, 3>{{2, 2, 3}});
  diff(std::array<int, 3>{{1, 2, 3}}, std::array<int, 3>{{1, 3, 3}});
  diff(std::array<int, 3>{{1, 2, 3}}, std::array<int, 3>{{1, 2, 4}});

  // sequence
  checkSequence<std::vector<int>>();
  checkSequence<std::deque<int>>();
  checkSequence<std::forward_list<int>>();
  checkSequence<std::list<int>>();
  checkSequence<std::set<int>>();
  checkSequence<std::multiset<int>>();

  // mapping
  checkCommonMapping<std::map<int, int>>();

  // unordered mapping
  checkCommonMapping<std::unordered_map<int, int>>();

  return 0;
}
