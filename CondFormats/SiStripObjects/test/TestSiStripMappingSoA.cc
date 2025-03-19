// A minimal test to ensure that
//   - sistrip::SiStripMappingSoA, sistrip::SiStripMappingHost can be compiled
//   - sistrip::SiStripMappingSoA can be allocated, modified and erased (on host)
//   - view-based element access works

// #include <cstdint>
#include "CondFormats/SiStripObjects/interface/SiStripMappingHost.h"

int main() {
  constexpr const int size = 42;
  SiStripMappingHost collection(size, cms::alpakatools::host());
  collection.zeroInitialise();

  const uint8_t arr[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  auto view = collection.view();
  for (uint32_t j = 0; j < size; j++) {
    view[j].input() = &arr[j % 10];
    view[j].inoff() = (size_t)j;
    view[j].offset() = (size_t)j;
    view[j].length() = (uint16_t)(j % 65536);
    view[j].fedID() = (uint16_t)(j % 65536);
    view[j].fedCh() = (uint8_t)(j % 256);
    view[j].detID() = 3 * j;
  }

  for (uint32_t j = 0; j < size; j++) {
    assert(view[j].input() == &arr[j % 10]);
    assert(view[j].inoff() == (size_t)j);
    assert(view[j].offset() == (size_t)j);
    assert(view[j].length() == (uint16_t)(j % 65536));
    assert(view[j].fedID() == (uint16_t)(j % 65536));
    assert(view[j].fedCh() == (uint8_t)(j % 256));
    assert(view[j].detID() == 3 * j);
  }

  return 0;
}