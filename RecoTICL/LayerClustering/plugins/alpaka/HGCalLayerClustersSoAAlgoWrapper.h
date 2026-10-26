#ifndef RecoLocalCalo_HGCalRecProducers_plugins_alpaka_HGCalLayerClustersSoAAlgoWrapper_h
#define RecoLocalCalo_HGCalRecProducers_plugins_alpaka_HGCalLayerClustersSoAAlgoWrapper_h

#include <cstdint>

#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsExtraDeviceCollection.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class HGCalLayerClustersSoAAlgoWrapper {
  public:
    // Fills the nLayerClusters layer clusters (energy, number of cells, seed, position and time) from the rechits and
    // their clustering (the layer cluster and the seed flag of each rechit). A layer cluster needs at least nHitsTime
    // rechits with a time to get a time.
    void run(Queue& queue,
             uint32_t nLayerClusters,
             float thresholdW0,
             float positionDeltaRho2,
             bool isScintillator,
             uint32_t nHitsTime,
             HGCalSoARecHitsDeviceCollection::ConstView recHits,
             HGCalSoARecHitsExtraDeviceCollection::ConstView recHitsClustering,
             reco::CaloClusterDeviceCollection::View layerClusters) const;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalCalo_HGCalRecProducers_plugins_alpaka_HGCalLayerClustersSoAAlgoWrapper_h
