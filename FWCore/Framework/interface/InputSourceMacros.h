#ifndef FWCore_Framework_InputSourceMacros_h
#define FWCore_Framework_InputSourceMacros_h

#include "FWCore/Framework/src/InputSourceFactory.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerPluginFactory.h"

#define DEFINE_FWK_INPUT_SOURCE(type) \
  DEFINE_EDM_PLUGIN (edm::InputSourcePluginFactory,type,#type); DEFINE_FWK_PSET_DESC_FILLER(type)

#endif
