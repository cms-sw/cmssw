#ifndef __RecoHGCal_TICL_alpaka_PatternRecognitionByCLUE3DSoA_H__
#define __RecoHGCal_TICL_alpaka_PatternRecognitionByCLUE3DSoA_H__

#include "RecoHGCal/TICL/interface/PatternRecognitionAlgoBaseSoA.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

    template <typename TILES>
    class PatternRecognitionByCLUE3DSoA final : public PatternRecognitionAlgoBaseSoAT<TILES> {
        public:
            PatternRecognitionByCLUE3DSoA(
                const edm::ParameterSet& config,
                edm::ConsumesCollector iC)
            : PatternRecognitionAlgoBaseSoAT<TILES>(config, iC) {};

            // ~PatternRecognitionbyCLUE3DSoA() override = default;

            void makeTracksters(
                Queue& queue,
                const typename PatternRecognitionAlgoBaseSoAT<TILES>::Inputs& inputs
            ) override;
    };
}

#endif