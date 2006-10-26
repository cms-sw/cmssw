#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <CalibCalorimetry/HcalStandardModules/interface/HcalPedestalAnalyzer.h>
#include <CalibCalorimetry/HcalStandardModules/interface/HcalLedAnalyzer.h>

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalPedestalAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(HcalLedAnalyzer);
