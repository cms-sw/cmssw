#ifndef RecoLocalCalo_HGCalRecProducers_plugins_alpaka_HGCalCLUEsteringAlgoWrapper_h
#define RecoLocalCalo_HGCalRecProducers_plugins_alpaka_HGCalCLUEsteringAlgoWrapper_h

#include <cstdint>
#include <span>

#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsExtraDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class HGCalCLUEsteringAlgoWrapper {
  public:
    // Clusters the rechits of each layer (one batch per layer, of batchItemSizes rechits) with CLUEstering: writes
    // the layer cluster and the seed flag of each rechit, and the number of layer clusters
    void run(Queue& queue,
             uint32_t nRecHits,
             float dc,
             float kappa,
             float outlierDeltaFactor,
             bool isScintillator,
             std::span<const uint32_t> batchItemSizes,
             HGCalSoARecHitsDeviceCollection::ConstView recHits,
             HGCalSoARecHitsExtraDeviceCollection::View recHitsClustering) const;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalCalo_HGCalRecProducers_plugins_alpaka_HGCalCLUEsteringAlgoWrapper_h
