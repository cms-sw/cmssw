#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "HcalDbProducer.h"
#include "HcalDbSource.h"
//#include "HcalDbSourcePool.h"
//Frontier #include "HcalDbSourceFrontier.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(HcalDbProducer)
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(HcalDbSource)
//DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(HcalDbSourcePool)
//Frontier DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(HcalDbSourceFrontier)
