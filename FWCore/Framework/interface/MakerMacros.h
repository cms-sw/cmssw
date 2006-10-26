#ifndef Framework_MakerMacros_h
#define Framework_MakerMacros_h

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/src/Factory.h"
#include "FWCore/Framework/src/WorkerMaker.h"

#if __GNUC_PREREQ (3,4)

#define DEFINE_FWK_MODULE(type) \
  DEFINE_SEAL_MODULE (); \
  DEFINE_SEAL_PLUGIN (edm::Factory,edm::WorkerMaker<type>,#type)

#define DEFINE_ANOTHER_FWK_MODULE(type) \
  DEFINE_SEAL_PLUGIN (edm::Factory,edm::WorkerMaker<type>,#type)

#else

#define DEFINE_FWK_MODULE(type) \
  DEFINE_SEAL_MODULE (); \
  DEFINE_SEAL_PLUGIN (edm::Factory,edm::WorkerMaker<type>,#type);

#define DEFINE_ANOTHER_FWK_MODULE(type) \
  DEFINE_SEAL_PLUGIN (edm::Factory,edm::WorkerMaker<type>,#type);

#endif

#endif
