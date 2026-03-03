#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoFactory.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceByPFN.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceByDNN.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginFactoryMacros.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"

EDM_REGISTER_VALIDATED_PLUGINFACTORY(TracksterInferenceAlgoFactory, "TracksterInferenceAlgoFactory");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterInferenceAlgoFactory, ticl::TracksterInferenceByPFN, "TracksterInferenceByPFN");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterInferenceAlgoFactory, ticl::TracksterInferenceByDNN, "TracksterInferenceByDNN");
