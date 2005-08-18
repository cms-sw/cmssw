#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "HcalDbProducer.h"
#include "HcalDbSourceHardcode.h"
#include "HcalDbSourceFrontier.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(HcalDbProducer)
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(HcalDbSourceHardcode)
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(HcalDbSourceFrontier)
