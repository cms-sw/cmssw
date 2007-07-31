#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserCorrectionService.h"

DEFINE_SEAL_MODULE();
DEFINE_FWK_EVENTSETUP_MODULE(EcalLaserCorrectionService);
