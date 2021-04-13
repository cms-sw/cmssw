#include "HeterogeneousCore/CUDAUtilities/interface/FlexiStorage.h"

#include <cassert>

using namespace cms::cuda;

int main() {
  FlexiStorage<int, 1024> a;

  assert(a.capacity() == 1024);

  FlexiStorage<int, -1> v;

  v.init(a.data(), 20);

  assert(v.capacity() == 20);

  assert(v.data() == a.data());

  a[4] = 42;

  assert(42 == a[4]);
  assert(42 == v[4]);

  auto const& ac = a;
  auto const& vc = v;

  assert(42 == ac[4]);
  assert(42 == vc[4]);

  assert(ac.data() == vc.data());

  return 0;
};
