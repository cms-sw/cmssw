#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoFactory.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceByDNN.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceByANN.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceByCNNv4.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginFactoryMacros.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"

EDM_REGISTER_VALIDATED_PLUGINFACTORY(TracksterInferenceAlgoFactory, "TracksterInferenceAlgoFactory");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterInferenceAlgoFactory, ticl::TracksterInferenceByDNN, "TracksterInferenceByDNN");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterInferenceAlgoFactory, ticl::TracksterInferenceByANN, "TracksterInferenceByANN");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterInferenceAlgoFactory,
                            ticl::TracksterInferenceByCNNv4,
                            "TracksterInferenceByCNNv4");
