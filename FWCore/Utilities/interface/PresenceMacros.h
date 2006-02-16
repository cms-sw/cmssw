#ifndef FWCore_Utilities_PresenceMacros_h
#define FWCore_Utilities_PresenceMacros_h

#include "PluginManager/ModuleDef.h"
#include "FWCore/Utilities/interface/PresenceFactory.h"
#include "FWCore/Utilities/interface/Presence.h"

#define DEFINE_FWK_PRESENCE(type) \
  DEFINE_SEAL_MODULE (); \
  DEFINE_SEAL_PLUGIN (edm::PresenceFactory,type,#type);

#define DEFINE_ANOTHER_FWK_PRESENCE(type) \
  DEFINE_SEAL_PLUGIN (edm::PresenceFactory,type,#type);

#endif
