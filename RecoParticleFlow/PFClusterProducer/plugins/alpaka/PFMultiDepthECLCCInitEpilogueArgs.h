#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCInitEpilogueArgs_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCInitEpilogueArgs_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringCCLabelsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthECLCCEpilogueArgsDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace ::cms::alpakatools;
  using namespace ::cms::alpakaintrinsics;

  class ECLCCInitEpilogueArgsKernel {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc,
        reco::PFMultiDepthECLCCEpilogueArgsDeviceCollection::View args,
        const reco::PFMultiDepthClusteringCCLabelsDeviceCollection::ConstView pfClusteringCCLabels) const {
      const unsigned int nVertices = pfClusteringCCLabels.size();

      if (::cms::alpakatools::once_per_grid(acc)) {
        args.blockCount() = 0;
      }

      for (int v : ::cms::alpakatools::uniform_elements(acc, nVertices)) {
        args[v].ccRHFOffset() = 0;
        args[v].ccRHFSize() = 0;
        args[v].rootMap() = 0;
        args[v].rootLocalMap() = 0;
        args[v].blockRHFOffset() = 0;
        args[v].ccEnergySeed() = 0;
        args[v].vertexMask() = ~0u;  // set bits to 1
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
