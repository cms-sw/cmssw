#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoTauTag/ImpactParameter/interface/ImpactParameter.h"
#include "RecoTauTag/ImpactParameter/interface/TauImpactParameterTest.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(ImpactParameter)
#DEFINE_ANOTHER_FWK_MODULE(TauImpactParameterTest)



