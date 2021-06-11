#include "FWCore/PluginManager/interface/PresenceMacros.h"
#include "MessageLogger.h"
#include "FWCore/MessageService/interface/SingleThreadMSPresence.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#pragma GCC visibility push(hidden)
using edm::service::MessageLogger;
using edm::service::SingleThreadMSPresence;

using MessageLoggerMaker = edm::serviceregistry::AllArgsMaker<edm::MessageLogger, MessageLogger>;
DEFINE_FWK_SERVICE_MAKER(MessageLogger, MessageLoggerMaker);

DEFINE_FWK_PRESENCE(SingleThreadMSPresence);
#pragma GCC visibility pop
