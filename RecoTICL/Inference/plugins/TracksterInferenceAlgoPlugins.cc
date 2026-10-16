#include "RecoTICL/Inference/interface/TracksterInferenceAlgoFactory.h"
#include "RecoTICL/Inference/interface/TracksterInferenceByPFN.h"
#include "RecoTICL/Inference/interface/TracksterInferenceByDNN.h"
#include "RecoTICL/Inference/interface/TracksterInferenceByCNN.h"
#include "RecoTICL/Inference/interface/TracksterInferenceByShowerShape.h"

#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"

DEFINE_EDM_VALIDATED_PLUGIN(TracksterInferenceAlgoFactory, ticl::TracksterInferenceByPFN, "TracksterInferenceByPFN");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterInferenceAlgoFactory, ticl::TracksterInferenceByDNN, "TracksterInferenceByDNN");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterInferenceAlgoFactory, ticl::TracksterInferenceByCNN, "TracksterInferenceByCNN");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterInferenceAlgoFactory,
                            ticl::TracksterInferenceByShowerShape,
                            "TracksterInferenceByShowerShape");
