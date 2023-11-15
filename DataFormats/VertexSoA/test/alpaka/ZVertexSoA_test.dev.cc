#include "DataFormats/VertexSoA/interface/alpaka/ZVertexSoACollection.h"
#include "DataFormats/VertexSoA/interface/ZVertexDevice.h"
#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"  // Check if this is really needed; code doesn't compile without it

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace alpaka;
  using namespace cms::alpakatools;

  namespace testZVertexSoAT {

    class TestFillKernel {
    public:
      template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(TAcc const& acc, reco::ZVertexSoAView zvertex_view) const {
        if (cms::alpakatools::once_per_grid(acc)) {
          zvertex_view.nvFinal() = 420;
        }

        for (int32_t j : elements_with_stride(acc, zvertex_view.metadata().size())) {
          zvertex_view[j].idv() = (int16_t)j;
          zvertex_view[j].zv() = (float)j;
          zvertex_view[j].wv() = (float)j;
          zvertex_view[j].chi2() = (float)j;
          zvertex_view[j].ptv2() = (float)j;
          zvertex_view[j].ndof() = (int32_t)j;
          zvertex_view[j].sortInd() = (uint16_t)j;
        }
      }
    };

    class TestVerifyKernel {
    public:
      template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(TAcc const& acc, reco::ZVertexSoAView zvertex_view) const {
        if (cms::alpakatools::once_per_grid(acc)) {
          ALPAKA_ASSERT_OFFLOAD(zvertex_view.nvFinal() == 420);
        }

        for (int32_t j : elements_with_stride(acc, zvertex_view.nvFinal())) {
          assert(zvertex_view[j].idv() == j);
          assert(zvertex_view[j].zv() - (float)j < 0.0001);
          assert(zvertex_view[j].wv() - (float)j < 0.0001);
          assert(zvertex_view[j].chi2() - (float)j < 0.0001);
          assert(zvertex_view[j].ptv2() - (float)j < 0.0001);
          assert(zvertex_view[j].ndof() == j);
          assert(zvertex_view[j].sortInd() == uint32_t(j));
        }
      }
    };

    void runKernels(reco::ZVertexSoAView zvertex_view, Queue& queue) {
      uint32_t items = 64;
      uint32_t groups = divide_up_by(zvertex_view.metadata().size(), items);
      auto workDiv = make_workdiv<Acc1D>(groups, items);
      alpaka::exec<Acc1D>(queue, workDiv, TestFillKernel{}, zvertex_view);
      alpaka::exec<Acc1D>(queue, workDiv, TestVerifyKernel{}, zvertex_view);
    }

  }  // namespace testZVertexSoAT

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
