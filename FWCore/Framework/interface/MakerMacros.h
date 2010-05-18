#ifndef Framework_MakerMacros_h
#define Framework_MakerMacros_h

#include "FWCore/Framework/src/Factory.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerPluginFactory.h"
// The following includes are temporary until a better
// solution can be found.  Placing these includes here
// leads to more physical coupling than is probably necessary.
// Another solution is to build a typeid lookup table in the
// implementation file (one every for each XXXWorker) and
// then include all the relevent worker headers in the
// implementation file only.
#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/src/OutputWorker.h"


#define DEFINE_FWK_MODULE(type) \
  DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<type>,#type); DEFINE_FWK_PSET_DESC_FILLER(type)
#endif
