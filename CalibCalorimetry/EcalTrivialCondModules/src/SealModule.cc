#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"

#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialObjectAnalyzer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(EcalTrivialConditionRetriever);
DEFINE_ANOTHER_FWK_MODULE(EcalTrivialObjectAnalyzer);
