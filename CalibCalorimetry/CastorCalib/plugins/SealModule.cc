#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "CastorDbProducer.h"
#include "CastorHardcodeCalibrations.h"
#include "CastorTextCalibrations.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(CastorDbProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CastorHardcodeCalibrations);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CastorTextCalibrations);
