#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "CastorDbProducer.h"
#include "CastorHardcodeCalibrations.h"
#include "CastorTextCalibrations.h"


DEFINE_FWK_EVENTSETUP_MODULE(CastorDbProducer);
DEFINE_FWK_EVENTSETUP_SOURCE(CastorHardcodeCalibrations);
DEFINE_FWK_EVENTSETUP_SOURCE(CastorTextCalibrations);
