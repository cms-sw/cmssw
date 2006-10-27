#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "HcalDbProducer.h"
#include "HcalHardcodeCalibrations.h"
#include "HcalTextCalibrations.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(HcalDbProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(HcalHardcodeCalibrations);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(HcalTextCalibrations);
