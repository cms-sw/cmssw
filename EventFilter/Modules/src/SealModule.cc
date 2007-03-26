#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IOPool/Streamer/interface/StreamerOutputModule.h"
#include "EventFilter/Modules/src/FUShmOutputModule.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

typedef edm::StreamerOutputModule<edm::FUShmOutputModule> ShmStreamConsumer;

using edm::FUShmOutputModule;
using namespace edm::serviceregistry;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(ShmStreamConsumer);
