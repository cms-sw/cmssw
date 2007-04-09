#ifndef Framework_InputSourceMacros_h
#define Framework_InputSourceMacros_h

#include "FWCore/Framework/src/InputSourceFactory.h"
#include "FWCore/Framework/interface/InputSource.h"

#define DEFINE_FWK_INPUT_SOURCE(type) \
  DEFINE_EDM_PLUGIN (edm::InputSourcePluginFactory,type,#type)

#define DEFINE_ANOTHER_FWK_INPUT_SOURCE(type) \
  DEFINE_EDM_PLUGIN (edm::InputSourcePluginFactory,type,#type)

#endif
