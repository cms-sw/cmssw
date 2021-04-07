#include "RecoHGCal/TICL/interface/PatternRecognitionPluginFactory.h"
#include "PatternRecognitionbyCA.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginFactoryMacros.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"

EDM_REGISTER_PLUGINFACTORY(PatternRecognitionFactory, "PatternRecognitionFactory");
EDM_REGISTER_PLUGINFACTORY(PatternRecognitionHFNoseFactory, "PatternRecognitionHFNoseFactory");
DEFINE_EDM_PLUGIN(PatternRecognitionFactory, ticl::PatternRecognitionbyCA<TICLLayerTiles>, "CA");
DEFINE_EDM_PLUGIN(PatternRecognitionHFNoseFactory, ticl::PatternRecognitionbyCA<TICLLayerTilesHFNose>, "CA");
