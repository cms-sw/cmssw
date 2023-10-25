// A minimal test to ensure that
//   - portabletest::TestSoA can be compiled
//   - portabletest::TestHostCollection can be allocated
//   - view-based element access works

#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

int main() {
  constexpr const int size = 42;
  portabletest::TestHostCollection collection(size, cms::alpakatools::host());

  const portabletest::Matrix matrix{{1, 2, 3, 4, 5, 6}, {2, 4, 6, 8, 10, 12}, {3, 6, 9, 12, 15, 18}};
  const portabletest::Array flags = {{6, 4, 2, 0}};

  collection.view().r() = 1.;

  for (int i = 0; i < size; ++i) {
    collection.view()[i] = {0.568, 0.823, 0., i, flags, matrix * i};
  }

  return 0;
}
