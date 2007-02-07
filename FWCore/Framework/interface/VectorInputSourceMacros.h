#ifndef Framework_VectorInputSourceMacros_h
#define Framework_VectorInputSourceMacros_h

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/src/VectorInputSourceFactory.h"
#include "FWCore/Framework/interface/VectorInputSource.h"

#define DEFINE_FWK_VECTOR_INPUT_SOURCE(type) \
  DEFINE_SEAL_MODULE (); \
  DEFINE_SEAL_PLUGIN (edm::VectorInputSourceFactory,type,#type)

#define DEFINE_ANOTHER_FWK_VECTOR_INPUT_SOURCE(type) \
  DEFINE_SEAL_PLUGIN (edm::VectorInputSourceFactory,type,#type)

#endif
