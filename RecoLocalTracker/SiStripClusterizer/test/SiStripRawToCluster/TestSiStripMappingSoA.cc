// A minimal test to ensure that
//   - sistrip::SiStripMappingSoA, sistrip::SiStripMappingHost can be compiled
//   - sistrip::SiStripMappingHost can be allocated, modified and erased (on host)
//   - view-based element access works
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripMappingHost.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"

using namespace sistrip;

int main() {
  // The typical size of a SiStripMappingHost collection is of O(conditions). Assumed 20k as a reasonable number.
  constexpr const int size = 200000;
  SiStripMappingHost collection(size, cms::alpakatools::host());
  collection.zeroInitialise();

  auto view = collection.view();
  for (uint32_t j = 0; j < size; ++j) {
    view.fedID(j) = (uint16_t)(j % 65536);
    view.fedCh(j) = (uint8_t)(j % 256);
    view.detID(j) = 3 * j;
    //
    view.fedChOff(j) = j;
    view.inoff(j) = (size_t)j;
    view.offset(j) = (size_t)j;
    view.length(j) = (uint16_t)(j % 65536);
    //
    view.readoutMode(j) = FEDReadoutMode(j % 15);
    view.packetCode(j) = (uint8_t)(j % 255);
  }

  for (uint32_t j = 0; j < size; ++j) {
    assert(view.fedID(j) == (uint16_t)(j % 65536));
    assert(view.fedCh(j) == (uint8_t)(j % 256));
    assert(view.detID(j) == 3 * j);
    //
    assert(view.fedChOff(j) == j);
    assert(view.inoff(j) == (size_t)j);
    assert(view.offset(j) == (size_t)j);
    assert(view.length(j) == (uint16_t)(j % 65536));
    //
    assert(view.readoutMode(j) == FEDReadoutMode(j % 15));
    assert(view.packetCode(j) == (uint8_t)(j % 255));
  }

  return 0;
}
