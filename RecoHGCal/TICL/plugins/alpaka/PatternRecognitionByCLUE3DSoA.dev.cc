#include "RecoHGCal/TICL/plugins/alpaka/PatternRecognitionByCLUE3DSoA.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoAClustersDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

    class TracksterKernel {
        public:
            template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                HGCalSoAClustersDeviceCollectionConstView const layerClustersSoA
            ) const {}
    };

    template <typename TILES>
    void PatternRecognitionByCLUE3DSoA<TILES>::makeTracksters(
        Queue& queue,
        const typename PatternRecognitionAlgoBaseSoAT<TILES>::Inputs& inputs
    ) {
        const auto clusters_soa = inputs.clusters;
        uint32_t items = 64;
        uint32_t groups = cms::alpakatools::divide_up_by(clusters_soa.metadata().size(), items);
        auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
        alpaka::exec<Acc1D>(queue, workDiv, TracksterKernel{}, clusters_soa);
    }

}

template class ALPAKA_ACCELERATOR_NAMESPACE::PatternRecognitionByCLUE3DSoA<TICLLayerTiles>;