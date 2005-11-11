#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/src/MessageService.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using edm::service::MessageService;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_SERVICE(MessageService)
