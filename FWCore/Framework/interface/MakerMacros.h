#ifndef Framework_MakerMacros_h
#define Framework_MakerMacros_h

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/src/Factory.h"
#include "FWCore/Framework/src/WorkerMaker.h"

#define DEFINE_FWK_MODULE(type) \
  DEFINE_SEAL_MODULE (); \
  DEFINE_SEAL_PLUGIN (edm::Factory,edm::WorkerMaker<type>,#type)

#define DEFINE_ANOTHER_FWK_MODULE(type) \
  DEFINE_SEAL_PLUGIN (edm::Factory,edm::WorkerMaker<type>,#type)

#endif
