#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserCorrectionService.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(EcalLaserCorrectionService);
//DEFINE_ANOTHER_FWK_MODULE(EcalLaserDbService);
//DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(HcalHardcodeCalibrations);
//DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(HcalTextCalibrations);

