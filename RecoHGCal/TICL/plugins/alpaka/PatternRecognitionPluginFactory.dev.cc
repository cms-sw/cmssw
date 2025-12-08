#include "RecoHGCal/TICL/plugins/alpaka/PatternRecognitionPluginFactory.h"
#include "RecoHGCal/TICL/plugins/alpaka/PatternRecognitionByCLUEstering.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginFactoryMacros.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"

EDM_REGISTER_VALIDATED_PLUGINFACTORY(PatternRecognitionFactoryAlpaka, "PatternRecognitionFactoryAlpaka");
// EDM_REGISTER_VALIDATED_PLUGINFACTORY(PatternRecognitionHFNoseFactoryAlpaka, "PatternRecognitionHFNoseFactoryAlpaka");
DEFINE_EDM_VALIDATED_PLUGIN(PatternRecognitionFactoryAlpaka,
                            ALPAKA_ACCELERATOR_NAMESPACE::PatternRecognitionByCLUEstering,
                            "CLUEstering");
