#ifndef Framework_VectorInputSourceMacros_h
#define Framework_VectorInputSourceMacros_h

#include "FWCore/Framework/src/VectorInputSourceFactory.h"
#include "FWCore/Framework/interface/VectorInputSource.h"

#define DEFINE_FWK_VECTOR_INPUT_SOURCE(type) \
  DEFINE_EDM_PLUGIN (edm::VectorInputSourcePluginFactory,type,#type)

#define DEFINE_ANOTHER_FWK_VECTOR_INPUT_SOURCE(type) \
  DEFINE_EDM_PLUGIN (edm::VectorInputSourcePluginFactory,type,#type)

#endif
