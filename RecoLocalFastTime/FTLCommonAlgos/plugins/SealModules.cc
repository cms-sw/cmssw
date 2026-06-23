#include "BTLUncalibRecHitAlgo.h"
#include "ETLUncalibRecHitAlgo.h"
#include "MTDRecHitAlgo.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"
DEFINE_EDM_VALIDATED_PLUGIN(BTLUncalibratedRecHitAlgoFactory, BTLUncalibRecHitAlgo, "BTLUncalibRecHitAlgo");
DEFINE_EDM_VALIDATED_PLUGIN(ETLUncalibratedRecHitAlgoFactory, ETLUncalibRecHitAlgo, "ETLUncalibRecHitAlgo");
DEFINE_EDM_VALIDATED_PLUGIN(MTDRecHitAlgoFactory, MTDRecHitAlgo, "MTDRecHitAlgo");
