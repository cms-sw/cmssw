#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "HcalDbProducer.h"
#include "HcalHardcodeCalibrations.h"
#include "HcalTextCalibrations.h"


DEFINE_FWK_EVENTSETUP_MODULE(HcalDbProducer);
DEFINE_FWK_EVENTSETUP_SOURCE(HcalHardcodeCalibrations);
DEFINE_FWK_EVENTSETUP_SOURCE(HcalTextCalibrations);
