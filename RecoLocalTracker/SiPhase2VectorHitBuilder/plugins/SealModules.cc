
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalTracker/SiPhase2VectorHitBuilder/interface/VectorHitBuilderEDProducer.h"
#include "RecoLocalTracker/SiPhase2VectorHitBuilder/plugins/SiPhase2RecHitMatcherESProducer.h"

DEFINE_FWK_EVENTSETUP_MODULE(SiPhase2RecHitMatcherESProducer);
DEFINE_FWK_MODULE(VectorHitBuilderEDProducer);

