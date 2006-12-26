#ifndef Framework_InputSourceMacros_h
#define Framework_InputSourceMacros_h

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/src/InputSourceFactory.h"
#include "FWCore/Framework/interface/InputSource.h"

#define DEFINE_FWK_INPUT_SOURCE(type) \
  DEFINE_SEAL_MODULE (); \
  DEFINE_SEAL_PLUGIN (edm::InputSourceFactory,type,#type)

#define DEFINE_ANOTHER_FWK_INPUT_SOURCE(type) \
  DEFINE_SEAL_PLUGIN (edm::InputSourceFactory,type,#type)

#endif
