#include "RecoHGCal/TICL/plugins/PatternRecognitionPluginFactory.h"
#include "PatternRecognitionbyCA.h"
#include "PatternRecognitionbyCLUE3D.h"
#include "PatternRecognitionbyFastJet.h"
#include "PatternRecognitionbyRecovery.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginFactoryMacros.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"

EDM_REGISTER_VALIDATED_PLUGINFACTORY(PatternRecognitionFactory, "PatternRecognitionFactory");
EDM_REGISTER_VALIDATED_PLUGINFACTORY(PatternRecognitionHFNoseFactory, "PatternRecognitionHFNoseFactory");
EDM_REGISTER_VALIDATED_PLUGINFACTORY(PatternRecognitionBarrelFactory, "PatternRecognitionBarrelFactory");

DEFINE_EDM_VALIDATED_PLUGIN(PatternRecognitionFactory, ticl::PatternRecognitionbyCA<TICLLayerTiles>, "CA");
DEFINE_EDM_VALIDATED_PLUGIN(PatternRecognitionFactory, ticl::PatternRecognitionbyCLUE3D<TICLLayerTiles>, "CLUE3D");
DEFINE_EDM_VALIDATED_PLUGIN(PatternRecognitionFactory, ticl::PatternRecognitionbyFastJet<TICLLayerTiles>, "FastJet");
DEFINE_EDM_VALIDATED_PLUGIN(PatternRecognitionFactory, ticl::PatternRecognitionbyRecovery<TICLLayerTiles>, "Recovery");

// Barrel
DEFINE_EDM_VALIDATED_PLUGIN(PatternRecognitionBarrelFactory,
                            ticl::PatternRecognitionbyCLUE3D<TICLLayerTilesBarrel>,
                            "CLUE3D");
DEFINE_EDM_VALIDATED_PLUGIN(PatternRecognitionBarrelFactory,
                            ticl::PatternRecognitionbyFastJet<TICLLayerTilesBarrel>,
                            "FastJet");

// HFNose
DEFINE_EDM_VALIDATED_PLUGIN(PatternRecognitionHFNoseFactory, ticl::PatternRecognitionbyCA<TICLLayerTilesHFNose>, "CA");
