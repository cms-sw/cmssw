#include "FWCore/PluginManager/interface/PresenceMacros.h"
#include "FWCore/MessageService/interface/MessageLogger.h"
#include "FWCore/MessageService/interface/MessageServicePresence.h"
#include "FWCore/MessageService/interface/SingleThreadMSPresence.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#pragma GCC visibility push(hidden)
using edm::service::MessageLogger;
using edm::service::MessageServicePresence;
using edm::service::SingleThreadMSPresence;
DEFINE_FWK_SERVICE(MessageLogger);
DEFINE_FWK_PRESENCE(MessageServicePresence);
DEFINE_FWK_PRESENCE(SingleThreadMSPresence);
#pragma GCC visibility pop
 
 
