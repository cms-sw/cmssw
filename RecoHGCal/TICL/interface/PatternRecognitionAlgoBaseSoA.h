#ifndef RecoHGCal_TICL_PatternRecognitionAlgoBaseSoA_H__
#define RecoHGCal_TICL_PatternRecognitionAlgoBaseSoA_H__

#include <memory>
#include <vector>
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/TICLSeedingRegion.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoAClustersDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoAClustersFilteredMaskDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
    template <typename TILES>
    class PatternRecognitionAlgoBaseSoAT {
        public:
            PatternRecognitionAlgoBaseSoAT(const edm::ParameterSet& config, edm::ConsumesCollector) {}
            virtual ~PatternRecognitionAlgoBaseSoAT() {};

            struct Inputs {
                const HGCalSoAClustersDeviceCollectionConstView& clusters;
                const HGCalSoAClustersFilteredMaskDeviceCollectionConstView& ms;
                const TILES& tiles;
                const std::vector<TICLSeedingRegion>& regions;

                Inputs(
                    const HGCalSoAClustersDeviceCollectionConstView& lC,
                    const HGCalSoAClustersFilteredMaskDeviceCollectionConstView& mS,
                    const TILES& tL,
                    const std::vector<TICLSeedingRegion>& rG)
                : clusters(lC), ms(mS), tiles(tL), regions(rG) {}
            };

            virtual void makeTracksters(
                Queue& queue,
                const Inputs& inputs
            );

    };
}


#endif