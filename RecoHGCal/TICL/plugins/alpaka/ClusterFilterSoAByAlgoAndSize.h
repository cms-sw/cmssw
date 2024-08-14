#ifndef RecoHGCal_TICL_alpaka_ClusterFilterSoAByAlgoAndSize_H__
#define RecoHGCal_TICL_alpaka_ClusterFilterSoAByAlgoAndSize_H__

#include <memory>
#include <utility>

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"

#include "DataFormats/HGCalReco/interface/Common.h"
// #include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoAClustersDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoAClustersFilteredMaskDeviceCollection.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {
    
    class ClusterFilterSoAByAlgoAndSize {
        public:
            ClusterFilterSoAByAlgoAndSize(const edm::ParameterSet& config) {}
            ~ClusterFilterSoAByAlgoAndSize() {}

            void filter(
                Queue& queue,
                const HGCalSoAClustersDeviceCollectionConstView layerClusters,
                HGCalSoAClustersFilteredMaskDeviceCollectionView layerClustersMask,
                int mix_cluster_size,
                int max_cluster_size
            );

        private:

    };
}

#endif