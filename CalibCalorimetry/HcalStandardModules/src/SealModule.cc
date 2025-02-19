#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <CalibCalorimetry/HcalStandardModules/interface/HFLightCal.h>
#include <CalibCalorimetry/HcalStandardModules/interface/HFPreLightCal.h>
#include <CalibCalorimetry/HcalStandardModules/interface/HFLightCalRand.h>

#include <CalibCalorimetry/HcalStandardModules/interface/HcalPedestalAnalyzer.h>
#include <CalibCalorimetry/HcalStandardModules/interface/HcalLedAnalyzer.h>


DEFINE_FWK_MODULE(HcalPedestalAnalyzer);
DEFINE_FWK_MODULE(HcalLedAnalyzer);

DEFINE_FWK_MODULE(HFLightCal);
DEFINE_FWK_MODULE(HFPreLightCal);
DEFINE_FWK_MODULE(HFLightCalRand);
