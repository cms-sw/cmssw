#ifndef FWCore_PluginManager_PresenceMacros_h
#define FWCore_PluginManager_PresenceMacros_h

#include "FWCore/PluginManager/interface/PresenceFactory.h"

#define DEFINE_FWK_PRESENCE(type) \
  DEFINE_EDM_PLUGIN (edm::PresencePluginFactory,type,#type)

#endif
