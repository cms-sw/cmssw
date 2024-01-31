#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigiErrorsSoACollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsHost.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"

using namespace alpaka;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  namespace testDigisSoA {

    class TestFillKernel {
    public:
      template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(TAcc const& acc, SiPixelDigiErrorsSoAView digiErrors_view) const {
        for (uint32_t j : elements_with_stride(acc, digiErrors_view.metadata().size())) {
          digiErrors_view[j].pixelErrors().rawId = j;
          digiErrors_view[j].pixelErrors().word = j;
          digiErrors_view[j].pixelErrors().errorType = j;
          digiErrors_view[j].pixelErrors().fedId = j;
        }
      }
    };

    class TestVerifyKernel {
    public:
      template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(TAcc const& acc, SiPixelDigiErrorsSoAConstView digiErrors_view) const {
        for (uint32_t j : elements_with_stride(acc, digiErrors_view.metadata().size())) {
          assert(digiErrors_view[j].pixelErrors().rawId == j);
          assert(digiErrors_view[j].pixelErrors().word == j);
          assert(digiErrors_view[j].pixelErrors().errorType == j % 256);
          assert(digiErrors_view[j].pixelErrors().fedId == j % 256);
        }
      }
    };

    void runKernels(SiPixelDigiErrorsSoAView digiErrors_view, Queue& queue) {
      uint32_t items = 64;
      uint32_t groups = divide_up_by(digiErrors_view.metadata().size(), items);
      auto workDiv = make_workdiv<Acc1D>(groups, items);
      alpaka::exec<Acc1D>(queue, workDiv, TestFillKernel{}, digiErrors_view);
      alpaka::exec<Acc1D>(queue, workDiv, TestVerifyKernel{}, digiErrors_view);
    }

  }  // namespace testDigisSoA
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
