#ifndef EDM_ISMACROS_H
#define EDM_ISMACROS_H

#include "PluginManager/ModuleDef.h"
#include "FWCore/CoreFramework/src/InputServiceFactory.h"
#include "FWCore/CoreFramework/interface/InputService.h"

#define DEFINE_FWK_INPUT_SERVICE(type) \
  DEFINE_SEAL_MODULE (); \
  DEFINE_SEAL_PLUGIN (edm::InputServiceFactory,type,#type);

#define DEFINE_ANOTHER_FWK_INPUT_SERVICE(type) \
  DEFINE_SEAL_PLUGIN (edm::InputServiceFactory,type,#type);

#endif
