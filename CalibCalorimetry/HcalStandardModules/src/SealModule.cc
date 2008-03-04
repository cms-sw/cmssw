#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <CalibCalorimetry/HcalStandardModules/interface/HFLightCal.h>
#include <CalibCalorimetry/HcalStandardModules/interface/HFPreLightCal.h>
#include <CalibCalorimetry/HcalStandardModules/interface/HFLightCalRand.h>

#include <CalibCalorimetry/HcalStandardModules/interface/HcalPedestalAnalyzer.h>
#include <CalibCalorimetry/HcalStandardModules/interface/HcalLedAnalyzer.h>

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalPedestalAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(HcalLedAnalyzer);

DEFINE_ANOTHER_FWK_MODULE(HFLightCal);
DEFINE_ANOTHER_FWK_MODULE(HFPreLightCal);
DEFINE_ANOTHER_FWK_MODULE(HFLightCalRand);
