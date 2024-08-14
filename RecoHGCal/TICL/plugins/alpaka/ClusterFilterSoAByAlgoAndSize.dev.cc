#include "RecoHGCal/TICL/plugins/alpaka/ClusterFilterSoAByAlgoAndSize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    class FilterKernel {
        public:
            template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                HGCalSoAClustersDeviceCollectionConstView const layerClustersSoA,
                HGCalSoAClustersFilteredMaskDeviceCollectionView layerClustersMaskSoA,
                int min_cluster_size,
                int max_cluster_size
            ) const {
                for (int32_t i: cms::alpakatools::uniform_elements(acc, layerClustersSoA.metadata().size())) {
                    layerClustersMaskSoA[i].mask() = 1.;
                    DetId detid = DetId(layerClustersSoA[i].seed());
                    if (
                        layerClustersSoA[i].cells() > max_cluster_size or 
                        ( (layerClustersSoA[i].cells() < min_cluster_size) and 
                            (
                                detid.det() == DetId::HGCalEE or
                                detid.det() == DetId::HGCalHSi or
                                (detid.det() == DetId::Forward and detid.subdetId() == static_cast<int>(ForwardSubdetector::HFNose))
                            )
                        )
                    ) layerClustersMaskSoA[i].mask() = 0.;
                }
            }
    }; 

    void ClusterFilterSoAByAlgoAndSize::filter(
        Queue& queue,
        const HGCalSoAClustersDeviceCollectionConstView layerClusters,
        HGCalSoAClustersFilteredMaskDeviceCollectionView layerClustersMask,
        int min_cluster_size,
        int max_cluster_size
    ) { 
        uint32_t items = 64;
        uint32_t groups = cms::alpakatools::divide_up_by(layerClusters.metadata().size(), items);
        auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);

        alpaka::exec<Acc1D>(queue, workDiv, FilterKernel{}, layerClusters, layerClustersMask, min_cluster_size, max_cluster_size);
    }

} // namespace ALPAKA_ACCELERATOR_NAMESPACE