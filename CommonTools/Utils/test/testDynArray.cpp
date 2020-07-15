#include "CommonTools/Utils/interface/DynArray.h"

struct A {
  A() {}
  A(int ii) : i(ii) {}
  int i = -3;
  double k = 0.1;

  virtual ~A() {}
};

#include <cassert>
#include <iostream>
#include <queue>

int main(int s, char **) {
  using T = A;

  unsigned n = 4 * s;

  //  alignas(alignof(T)) unsigned char a_storage[sizeof(T)*n];
  //  DynArray<T> a(a_storage,n);

  declareDynArray(T, n, a);

  // T b[n];
  declareDynArray(T, n, b);

  b[0].i = 42;
  b[n - 1].i = -42;

  auto pa = [&](auto i) {
    a[1].k = 0.3;
    return a[i].k;
  };
  auto pb = [&](auto i) {
    b[1].k = 0.5;
    return b[i].k;
  };

  auto loop = [&](auto k) {
    for (auto const &q : a)
      k = std::min(k, q.k);
    return k;
  };

  std::cout << a[n - 1].k << ' ' << pa(1) << ' ' << loop(2.3) << std::endl;
  std::cout << b[n - 1].k << ' ' << pb(1) << std::endl;

  assert(b.back().i == -42);
  assert(b.front().i == 42);

  initDynArray(bool, n, q, true);
  if (q[n - 1])
    std::cout << "ok" << std::endl;

  auto sn = 2 * n;
  unInitDynArray(T, sn + n, c);
  assert(c.empty());
  for (int i = 0; i < int(sn); ++i)
    c.push_back(i);
  assert(c.size() == sn);
  assert(c.front().i == 0);
  assert(c.back().i == int(sn - 1));
  c[1].k = 3.14;

  a = std::move(c);

  assert(a.size() == sn);
  assert(c.empty());
  assert(a[1].k == 3.14);

  std::swap(a, b);
  assert(b.size() == sn);
  assert(a.size() == n);

  unInitDynArray(int, sn, qst);  // queue storage
  auto cmp = [](int i, int j) { return i < j; };
  std::priority_queue<int, DynArray<int>, decltype(cmp)> qq(cmp, std::move(qst));
  assert(qq.empty());
  for (int i = 0; i < int(sn); ++i)
    qq.push(i + 1);
  assert(qq.size() == sn);
  for (int i = 0; i < int(sn); ++i) {
    assert(qq.size() == sn - i);
    assert(qq.top() == int(sn - i));
    qq.pop();
  }

  assert(qq.empty());
  qq.push(3);
  qq.push(7);
  qq.push(-3);
  assert(qq.top() == 7);

  return 0;
};
