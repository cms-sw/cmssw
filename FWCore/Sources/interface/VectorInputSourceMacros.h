#ifndef FWCore_Sources_VectorInputSourceMacros_h
#define FWCore_Sources_VectorInputSourceMacros_h

#include "FWCore/Sources/interface/VectorInputSourceFactory.h"
#include "FWCore/Sources/interface/VectorInputSource.h"

#define DEFINE_FWK_VECTOR_INPUT_SOURCE(type) \
  DEFINE_EDM_PLUGIN (edm::VectorInputSourcePluginFactory,type,#type)

#endif
