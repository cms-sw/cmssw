#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "IOPool/Streamer/interface/StreamerOutputModule.h"
#include "EventFilter/Modules/src/FUShmOutputModule.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "EventFilter/Modules/interface/FUShmDQMOutputService.h"
#include "EventFilter/Modules/interface/ShmOutputModuleRegistry.h"

typedef edm::StreamerOutputModule<edm::FUShmOutputModule> ShmStreamConsumer;

using edm::FUShmOutputModule;
using namespace edm::serviceregistry;
using namespace evf;

typedef AllArgsMaker<FUShmDQMOutputService> dssMaker;
typedef ParameterSetMaker<ShmOutputModuleRegistry> maker3;


DEFINE_FWK_MODULE(ShmStreamConsumer);
DEFINE_FWK_SERVICE_MAKER(FUShmDQMOutputService,dssMaker);
DEFINE_FWK_SERVICE_MAKER(ShmOutputModuleRegistry,maker3);

