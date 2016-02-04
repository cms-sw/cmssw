#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"
#include "CalibCalorimetry/EcalTrivialCondModules/interface/ESTrivialConditionRetriever.h"

#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialObjectAnalyzer.h"


DEFINE_FWK_EVENTSETUP_SOURCE(EcalTrivialConditionRetriever);
DEFINE_FWK_EVENTSETUP_SOURCE(ESTrivialConditionRetriever);
DEFINE_FWK_MODULE(EcalTrivialObjectAnalyzer);
