#ifndef Framework_VectorInputSourceMacros_h
#define Framework_VectorInputSourceMacros_h

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/src/VectorInputSourceFactory.h"
#include "FWCore/Framework/interface/VectorInputSource.h"

#include "FWCore/Utilities/interface/GCCPrerequisite.h"

#if GCC_PREREQUISITE(3,4,4)

#define DEFINE_FWK_VECTOR_INPUT_SOURCE(type) \
  DEFINE_SEAL_MODULE (); \
  DEFINE_SEAL_PLUGIN (edm::VectorInputSourceFactory,type,#type)

#define DEFINE_ANOTHER_FWK_VECTOR_INPUT_SOURCE(type) \
  DEFINE_SEAL_PLUGIN (edm::VectorInputSourceFactory,type,#type)

#else

#define DEFINE_FWK_VECTOR_INPUT_SOURCE(type) \
  DEFINE_SEAL_MODULE (); \
  DEFINE_SEAL_PLUGIN (edm::VectorInputSourceFactory,type,#type);

#define DEFINE_ANOTHER_FWK_VECTOR_INPUT_SOURCE(type) \
  DEFINE_SEAL_PLUGIN (edm::VectorInputSourceFactory,type,#type);

#endif

#endif
