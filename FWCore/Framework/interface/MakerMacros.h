#ifndef Framework_MakerMacros_h
#define Framework_MakerMacros_h

#include "FWCore/Framework/src/Factory.h"
#include "FWCore/Framework/src/WorkerMaker.h"
// The following includes are temporary until a better
// solution can be found.  Placing these includes here
// leads to more physical coupling than is probably necessary.
// Another solution is to build a typeid lookup table in the
// implementation file (one every for each XXXWorker) and
// then include all the relevent worker headers in the
// implementation file only.
#include "FWCore/Framework/src/ProducerWorker.h"
#include "FWCore/Framework/src/FilterWorker.h"
#include "FWCore/Framework/src/AnalyzerWorker.h"
#include "FWCore/Framework/src/OutputWorker.h"


#define DEFINE_FWK_MODULE(type) \
  DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<type>,#type)

#define DEFINE_ANOTHER_FWK_MODULE(type) \
  DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<type>,#type)

// for backward comatibility
#include "FWCore/PluginManager/interface/ModuleDef.h"
#endif
