#ifndef Framework_SecondaryInputSourceMacros_h
#define Framework_SecondaryInputSourceMacros_h

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/src/SecondaryInputSourceFactory.h"
#include "FWCore/Framework/interface/SecondaryInputSource.h"

#define DEFINE_FWK_SECONDARY_INPUT_SOURCE(type) \
  DEFINE_SEAL_MODULE (); \
  DEFINE_SEAL_PLUGIN (edm::SecondaryInputSourceFactory,type,#type);

#define DEFINE_ANOTHER_FWK_SECONDARY_INPUT_SOURCE(type) \
  DEFINE_SEAL_PLUGIN (edm::SecondaryInputSourceFactory,type,#type);

#endif
