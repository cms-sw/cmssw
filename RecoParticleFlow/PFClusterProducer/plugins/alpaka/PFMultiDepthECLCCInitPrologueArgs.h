#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCInitPrologueArgs_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCInitPrologueArgs_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringCCLabelsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthECLCCPrologueArgsDeviceCollection.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace ::cms::alpakatools;
  using namespace ::cms::alpakaintrinsics;

  class ECLCCInitPrologueArgsKernel {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc,
        reco::PFMultiDepthECLCCPrologueArgsDeviceCollection::View args,
        const reco::PFMultiDepthClusteringCCLabelsDeviceCollection::ConstView pfClusteringCCLabels) const {

            const unsigned int nVertices = pfClusteringCCLabels.size();

            if (::cms::alpakatools::once_per_grid(acc)) {
                args.blockCount() = 0;
            }

            for (int v : ::cms::alpakatools::uniform_elements(acc, nVertices)) {;
              args[v].ccOffset() = 0;
              args[v].ccSize() = 0;
              args[v].blockInternCCSize() = 0;
            }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
