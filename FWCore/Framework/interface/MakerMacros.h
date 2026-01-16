#ifndef Framework_MakerMacros_h
#define Framework_MakerMacros_h

#include "FWCore/Framework/interface/maker/ModuleMakerPluginFactory.h"
#include "FWCore/Framework/interface/maker/ModuleMaker.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerPluginFactory.h"
// The following includes are temporary until a better
// solution can be found.  Placing these includes here
// leads to more physical coupling than is probably necessary.
// Another solution is to build a typeid lookup table in the
// implementation file (one every for each XXXWorker) and
// then include all the relevent worker headers in the
// implementation file only.
#include "FWCore/Framework/interface/maker/WorkerT.h"

#define DEFINE_FWK_MODULE(type)                                                    \
  DEFINE_EDM_PLUGIN(edm::ModuleMakerPluginFactory, edm::ModuleMaker<type>, #type); \
  DEFINE_FWK_PSET_DESC_FILLER(type)
#endif
