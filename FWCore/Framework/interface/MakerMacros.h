#ifndef EDM_MAKERMACROS_H
#define EDM_MAKERMACROS_H

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/src/Factory.h"
#include "FWCore/Framework/src/WorkerMaker.h"

#define DEFINE_FWK_MODULE(type) \
  DEFINE_SEAL_MODULE (); \
  DEFINE_SEAL_PLUGIN (edm::Factory,edm::WorkerMaker<type>,#type);

#define DEFINE_ANOTHER_FWK_MODULE(type) \
  DEFINE_SEAL_PLUGIN (edm::Factory,edm::WorkerMaker<type>,#type);

#endif
