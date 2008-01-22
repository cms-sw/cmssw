#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IOPool/Streamer/interface/StreamerOutputModule.h"
#include "EventFilter/Modules/src/FUShmOutputModule.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "EventFilter/Modules/interface/FUShmDQMOutputService.h"

typedef edm::StreamerOutputModule<edm::FUShmOutputModule> ShmStreamConsumer;

using edm::FUShmOutputModule;
using namespace edm::serviceregistry;

typedef AllArgsMaker<FUShmDQMOutputService> dssMaker;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(ShmStreamConsumer);
DEFINE_ANOTHER_FWK_SERVICE_MAKER(FUShmDQMOutputService,dssMaker);
