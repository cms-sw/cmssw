#include <alpaka/alpaka.hpp>

#include "DataFormats/VertexSoA/interface/ZVertexDevice.h"
#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/VertexSoA/interface/alpaka/ZVertexSoACollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testZVertexSoAT {

  class TestFillKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  reco::ZVertexSoAView zvertex_view,
                                  reco::ZVertexTracksSoAView ztracks_view) const {
      if (cms::alpakatools::once_per_grid(acc)) {
        zvertex_view.nvFinal() = 420;
      }

      for (int32_t j : cms::alpakatools::uniform_elements(acc, zvertex_view.metadata().size())) {
        zvertex_view[j].zv() = (float)j;
        zvertex_view[j].wv() = (float)j;
        zvertex_view[j].chi2() = (float)j;
        zvertex_view[j].ptv2() = (float)j;
        zvertex_view[j].sortInd() = (uint16_t)j;
      }
      for (int32_t j : cms::alpakatools::uniform_elements(acc, ztracks_view.metadata().size())) {
        ztracks_view[j].idv() = (int16_t)j;
        ztracks_view[j].ndof() = (int32_t)j;
      }
    }
  };

  class TestVerifyKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  reco::ZVertexSoAView zvertex_view,
                                  reco::ZVertexTracksSoAView ztracks_view) const {
      if (cms::alpakatools::once_per_grid(acc)) {
        ALPAKA_ASSERT_ACC(zvertex_view.nvFinal() == 420);
      }

      for (int32_t j : cms::alpakatools::uniform_elements(acc, zvertex_view.nvFinal())) {
        ALPAKA_ASSERT(zvertex_view[j].zv() - (float)j < 0.0001);
        ALPAKA_ASSERT(zvertex_view[j].wv() - (float)j < 0.0001);
        ALPAKA_ASSERT(zvertex_view[j].chi2() - (float)j < 0.0001);
        ALPAKA_ASSERT(zvertex_view[j].ptv2() - (float)j < 0.0001);
        ALPAKA_ASSERT(zvertex_view[j].sortInd() == uint32_t(j));
      }
      for (int32_t j : cms::alpakatools::uniform_elements(acc, ztracks_view.metadata().size())) {
        ALPAKA_ASSERT(ztracks_view[j].idv() == j);
        ALPAKA_ASSERT(ztracks_view[j].ndof() == j);
      }
    }
  };

  void runKernels(reco::ZVertexSoAView zvertex_view, reco::ZVertexTracksSoAView ztracks_view, Queue& queue) {
    uint32_t items = 64;
    uint32_t groups = cms::alpakatools::divide_up_by(zvertex_view.metadata().size(), items);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue, workDiv, TestFillKernel{}, zvertex_view, ztracks_view);
    alpaka::exec<Acc1D>(queue, workDiv, TestVerifyKernel{}, zvertex_view, ztracks_view);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testZVertexSoAT
