#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "CondFormats/SiStripObjects/interface/SiStripMappingSoA.h"
#include "TestSiStripMappingDevice.h"

using namespace alpaka;

namespace ALPAKA_ACCELERATOR_NAMESPACE::testMappingSoA {
  class TestFillKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, SiStripMappingView view) const {
      const uint8_t arr[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
      for (int32_t j : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
        view[j].input() = &arr[j % 10];
        view[j].inoff() = (size_t)j;
        view[j].offset() = (size_t)j;
        view[j].length() = (uint16_t)(j % 65536);
        view[j].fedID() = (uint16_t)(j % 65536);
        view[j].fedCh() = (uint8_t)(j % 256);
        view[j].detID() = 3 * j;
      }
    }
  };

  class TestVerifyKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, SiStripMappingConstView view) const {
      const uint8_t arr[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
      for (uint32_t j : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
        ALPAKA_ASSERT_ACC(view[j].input() - arr == j % 10);
        ALPAKA_ASSERT_ACC(view[j].inoff() == (size_t)j);
        ALPAKA_ASSERT_ACC(view[j].offset() == (size_t)j);
        ALPAKA_ASSERT_ACC(view[j].length() == (uint16_t)(j % 65536));
        ALPAKA_ASSERT_ACC(view[j].fedID() == (uint16_t)(j % 65536));
        ALPAKA_ASSERT_ACC(view[j].fedCh() == (uint8_t)(j % 256));
        ALPAKA_ASSERT_ACC(view[j].detID() == 3 * j);
      }
    }
  };

  void runKernels(SiStripMappingView view, Queue& queue) {
    uint32_t items = 64;
    uint32_t groups = cms::alpakatools::divide_up_by(view.metadata().size(), items);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue, workDiv, TestFillKernel{}, view);
    alpaka::exec<Acc1D>(queue, workDiv, TestVerifyKernel{}, view);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testMappingSoA