#ifndef Framework_MakerMacros_h
#define Framework_MakerMacros_h

#include "FWCore/Framework/src/Factory.h"
#include "FWCore/Framework/src/WorkerMaker.h"

#define DEFINE_FWK_MODULE(type) \
  DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<type>,#type)

#define DEFINE_ANOTHER_FWK_MODULE(type) \
  DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<type>,#type)

// for backward comatibility
#include "FWCore/PluginManager/interface/ModuleDef.h"
#endif
