#ifndef RecoLocalCalo_HGCalRecProducers_plugins_alpaka_HGCalLayerClustersSoAAlgoWrapper_h
#define RecoLocalCalo_HGCalRecProducers_plugins_alpaka_HGCalLayerClustersSoAAlgoWrapper_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsHostCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsExtraDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoAClustersDeviceCollection.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/alpaka/HGCalSoAClustersExtraDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class HGCalLayerClustersSoAAlgoWrapper {
  public:
    void run(Queue& queue,
             const unsigned int numer_of_clusters,
             float thresholdW0,
             float positionDeltaRho2,
             const HGCalSoARecHitsDeviceCollection::ConstView input_rechits_soa,
             const HGCalSoARecHitsExtraDeviceCollection::ConstView input_clusters_soa,
             HGCalSoAClustersDeviceCollection::View outputs,
             HGCalSoAClustersExtraDeviceCollection::View outputs_service) const;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalCalo_HGCalRecProducers_plugins_alpaka_HGCalLayerClustersSoAAlgoWrapper_h
