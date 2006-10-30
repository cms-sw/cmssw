#ifndef Framework_InputSourceMacros_h
#define Framework_InputSourceMacros_h

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/src/InputSourceFactory.h"
#include "FWCore/Framework/interface/InputSource.h"

#include "FWCore/Utilities/interface/GCCPrerequisite.h"

#if GCC_PREREQUISITE(3,4,4)

#define DEFINE_FWK_INPUT_SOURCE(type) \
  DEFINE_SEAL_MODULE (); \
  DEFINE_SEAL_PLUGIN (edm::InputSourceFactory,type,#type)

#define DEFINE_ANOTHER_FWK_INPUT_SOURCE(type) \
  DEFINE_SEAL_PLUGIN (edm::InputSourceFactory,type,#type)

#else

#define DEFINE_FWK_INPUT_SOURCE(type) \
  DEFINE_SEAL_MODULE (); \
  DEFINE_SEAL_PLUGIN (edm::InputSourceFactory,type,#type);

#define DEFINE_ANOTHER_FWK_INPUT_SOURCE(type) \
  DEFINE_SEAL_PLUGIN (edm::InputSourceFactory,type,#type);

#endif

#endif
