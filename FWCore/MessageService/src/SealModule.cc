#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/PresenceMacros.h"
#include "FWCore/MessageService/interface/MessageLogger.h"
#include "FWCore/MessageService/interface/MessageServicePresence.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using edm::service::MessageLogger;
using edm::service::MessageServicePresence;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_SERVICE(MessageLogger)
DEFINE_ANOTHER_FWK_PRESENCE(MessageServicePresence)
