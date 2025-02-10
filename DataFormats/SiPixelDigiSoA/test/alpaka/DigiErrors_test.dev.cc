#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsHost.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigiErrorsSoACollection.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "DigiErrors_test.h"

using namespace alpaka;

namespace ALPAKA_ACCELERATOR_NAMESPACE::testDigisSoA {

  class TestFillKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, SiPixelDigiErrorsSoAView digiErrors_view) const {
      for (uint32_t j : cms::alpakatools::uniform_elements(acc, digiErrors_view.metadata().size())) {
        digiErrors_view[j].pixelErrors().rawId = j;
        digiErrors_view[j].pixelErrors().word = j;
        digiErrors_view[j].pixelErrors().errorType = j;
        digiErrors_view[j].pixelErrors().fedId = j;
      }
    }
  };

  class TestVerifyKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, SiPixelDigiErrorsSoAConstView digiErrors_view) const {
      for (uint32_t j : cms::alpakatools::uniform_elements(acc, digiErrors_view.metadata().size())) {
        ALPAKA_ASSERT_ACC(digiErrors_view[j].pixelErrors().rawId == j);
        ALPAKA_ASSERT_ACC(digiErrors_view[j].pixelErrors().word == j);
        ALPAKA_ASSERT_ACC(digiErrors_view[j].pixelErrors().errorType == j % 256);
        ALPAKA_ASSERT_ACC(digiErrors_view[j].pixelErrors().fedId == j % 256);
      }
    }
  };

  void runKernels(SiPixelDigiErrorsSoAView digiErrors_view, Queue& queue) {
    uint32_t items = 64;
    uint32_t groups = cms::alpakatools::divide_up_by(digiErrors_view.metadata().size(), items);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue, workDiv, TestFillKernel{}, digiErrors_view);
    alpaka::exec<Acc1D>(queue, workDiv, TestVerifyKernel{}, digiErrors_view);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testDigisSoA
