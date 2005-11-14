#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/src/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using edm::service::MessageLogger;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_SERVICE(MessageLogger)
