#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripMappingSoA.h"

#include "TestSiStripMappingDevice.h"

using namespace alpaka;

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip::testMappingSoA {
  class TestFillKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, SiStripMappingView view) const {
      for (int32_t j : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
        view.detID(j) = 3 * j;
        view.fedID(j) = (uint16_t)(j % 65536);
        view.fedCh(j) = (uint16_t)(j % 256);
        //
        view.fedChOff(j) = j;
        view.inoff(j) = (size_t)j;
        view.offset(j) = (size_t)j;
        view.length(j) = (uint16_t)(j % 65536);
        //
        view.readoutMode(j) = FEDReadoutMode(j % 15);
        view.packetCode(j) = (uint8_t)(j % 255);
      }
    }
  };

  class TestVerifyKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, SiStripMappingConstView view) const {
      for (uint32_t j : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
        ALPAKA_ASSERT_ACC(view.detID(j) == 3 * j);
        ALPAKA_ASSERT_ACC(view.fedID(j) == (uint16_t)(j % 65536));
        ALPAKA_ASSERT_ACC(view.fedCh(j) == (uint16_t)(j % 256));
        //
        ALPAKA_ASSERT_ACC(view.fedChOff(j) == j);
        ALPAKA_ASSERT_ACC(view.inoff(j) == (size_t)j);
        ALPAKA_ASSERT_ACC(view.offset(j) == (size_t)j);
        ALPAKA_ASSERT_ACC(view.length(j) == (uint16_t)(j % 65536));
        //
        ALPAKA_ASSERT_ACC(view.readoutMode(j) == FEDReadoutMode(j % 15));
        ALPAKA_ASSERT_ACC(view.packetCode(j) == (uint8_t)(j % 255));
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

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip::testMappingSoA
