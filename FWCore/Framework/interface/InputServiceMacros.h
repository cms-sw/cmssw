#ifndef Framework_InputServiceMacros_h
#define Framework_InputServiceMacros_h

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/src/InputServiceFactory.h"
#include "FWCore/Framework/interface/InputService.h"

#define DEFINE_FWK_INPUT_SERVICE(type) \
  DEFINE_SEAL_MODULE (); \
  DEFINE_SEAL_PLUGIN (edm::InputServiceFactory,type,#type);

#define DEFINE_ANOTHER_FWK_INPUT_SERVICE(type) \
  DEFINE_SEAL_PLUGIN (edm::InputServiceFactory,type,#type);

#endif
