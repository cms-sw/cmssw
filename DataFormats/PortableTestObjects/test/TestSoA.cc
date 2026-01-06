// A minimal test to ensure that
//   - portabletest::TestSoA can be compiled
//   - portabletest::TestHostCollection can be allocated
//   - portabletest::TestHostCollection can be erased
//   - view-based element access works

#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

int main() {
  constexpr const int size = 42;
  constexpr const int size2 = 21;
  constexpr const int size3 = 69;
  portabletest::TestHostCollection collection(size, cms::alpakatools::host());

  const portabletest::Matrix matrix{{1, 2, 3, 4, 5, 6}, {2, 4, 6, 8, 10, 12}, {3, 6, 9, 12, 15, 18}};
  const portabletest::Array flags = {{6, 4, 2, 0}};

  collection.zeroInitialise();

  collection.view().r() = 1.;

  for (int i = 0; i < size; ++i) {
    collection.view()[i] = {0.568, 0.823, 0., i, flags, matrix * i};
  }

  // Same test but for portabletest::TestHostCollection2 --> collection with 2 SoABlocks

  portabletest::TestHostCollection2 collection2(cms::alpakatools::host(), size, size2);
  collection2.zeroInitialise();

  collection2.view().first().r() = 1.;
  collection2.view().second().r2() = 2.;

  for (int i = 0; i < size; ++i) {
    collection2.view().first()[i] = {0.568, 0.823, 0., i, flags, matrix * i};
  }

  for (int i = 0; i < size2; ++i) {
    collection2.view().second()[i] = {1.568, 1.823, 1., i + 1000, matrix * (i + 1000)};
  }

  // Same test but for portabletest::TestHostCollection3 --> collection with 3 SoABlocks

  portabletest::TestHostCollection3 collection3(cms::alpakatools::host(), size, size2, size3);
  collection3.zeroInitialise();

  collection3.view().first().r() = 1.;
  collection3.view().second().r2() = 2.;
  collection3.view().third().r3() = 3.;

  for (int i = 0; i < size; ++i) {
    collection3.view().first()[i] = {0.568, 0.823, 0., i, flags, matrix * i};
  }

  for (int i = 0; i < size2; ++i) {
    collection3.view().second()[i] = {1.568, 1.823, 1., i + 1000, matrix * (i + 1000)};
  }

  for (int i = 0; i < size3; ++i) {
    collection3.view().third()[i] = {2.568, 2.823, 2., i + 2000, matrix * (i + 2000)};
  }

  return 0;
}
