#include <alpaka/alpaka.hpp>

#include "DataFormats/VertexSoA/interface/ZVertexDevice.h"
#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/VertexSoA/interface/alpaka/ZVertexSoACollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testZVertexSoAT {

  class TestFillKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, reco::ZVertexBlocksView zvertex_blocks_view) const {
      if (cms::alpakatools::once_per_grid(acc)) {
        zvertex_blocks_view.zvertex().nvFinal() = 420;
      }

      for (int32_t j : cms::alpakatools::uniform_elements(acc, zvertex_blocks_view.zvertex().metadata().size())) {
        zvertex_blocks_view.zvertex()[j].zv() = (float)j;
        zvertex_blocks_view.zvertex()[j].wv() = (float)j;
        zvertex_blocks_view.zvertex()[j].chi2() = (float)j;
        zvertex_blocks_view.zvertex()[j].ptv2() = (float)j;
        zvertex_blocks_view.zvertex()[j].sortInd() = (uint16_t)j;
      }
      for (int32_t j : cms::alpakatools::uniform_elements(acc, zvertex_blocks_view.zvertexTracks().metadata().size())) {
        zvertex_blocks_view.zvertexTracks()[j].idv() = (int16_t)j;
        zvertex_blocks_view.zvertexTracks()[j].ndof() = (int32_t)j;
      }
    }
  };

  class TestVerifyKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, reco::ZVertexBlocksView zvertex_blocks_view) const {
      if (cms::alpakatools::once_per_grid(acc)) {
        ALPAKA_ASSERT_ACC(zvertex_blocks_view.zvertex().nvFinal() == 420);
      }

      for (int32_t j : cms::alpakatools::uniform_elements(acc, zvertex_blocks_view.zvertex().nvFinal())) {
        ALPAKA_ASSERT(zvertex_blocks_view.zvertex()[j].zv() - (float)j < 0.0001);
        ALPAKA_ASSERT(zvertex_blocks_view.zvertex()[j].wv() - (float)j < 0.0001);
        ALPAKA_ASSERT(zvertex_blocks_view.zvertex()[j].chi2() - (float)j < 0.0001);
        ALPAKA_ASSERT(zvertex_blocks_view.zvertex()[j].ptv2() - (float)j < 0.0001);
        ALPAKA_ASSERT(zvertex_blocks_view.zvertex()[j].sortInd() == uint32_t(j));
      }
      for (int32_t j : cms::alpakatools::uniform_elements(acc, zvertex_blocks_view.zvertexTracks().metadata().size())) {
        ALPAKA_ASSERT(zvertex_blocks_view.zvertexTracks()[j].idv() == j);
        ALPAKA_ASSERT(zvertex_blocks_view.zvertexTracks()[j].ndof() == j);
      }
    }
  };

  void runKernels(reco::ZVertexBlocksView zvertex_blocks_view, Queue& queue) {
    uint32_t items = 64;
    uint32_t groups = cms::alpakatools::divide_up_by(zvertex_blocks_view.metadata().maxSize(), items);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue, workDiv, TestFillKernel{}, zvertex_blocks_view);
    alpaka::exec<Acc1D>(queue, workDiv, TestVerifyKernel{}, zvertex_blocks_view);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testZVertexSoAT
