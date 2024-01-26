#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"

using namespace alpaka;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  namespace testDigisSoA {

    class TestFillKernel {
    public:
      template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(TAcc const& acc, SiPixelDigisSoAView digi_view) const {
        for (int32_t j : elements_with_stride(acc, digi_view.metadata().size())) {
          digi_view[j].clus() = j;
          digi_view[j].rawIdArr() = j * 2;
          digi_view[j].xx() = j * 3;
          digi_view[j].moduleId() = j * 4;
        }
      }
    };

    class TestVerifyKernel {
    public:
      template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(TAcc const& acc, SiPixelDigisSoAConstView digi_view) const {
        for (uint32_t j : elements_with_stride(acc, digi_view.metadata().size())) {
          assert(digi_view[j].clus() == int(j));
          assert(digi_view[j].rawIdArr() == j * 2);
          assert(digi_view[j].xx() == j * 3);
          assert(digi_view[j].moduleId() == j * 4);
        }
      }
    };

    void runKernels(SiPixelDigisSoAView digi_view, Queue& queue) {
      uint32_t items = 64;
      uint32_t groups = divide_up_by(digi_view.metadata().size(), items);
      auto workDiv = make_workdiv<Acc1D>(groups, items);
      alpaka::exec<Acc1D>(queue, workDiv, TestFillKernel{}, digi_view);
      alpaka::exec<Acc1D>(queue, workDiv, TestVerifyKernel{}, digi_view);
    }

  }  // namespace testDigisSoA
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
