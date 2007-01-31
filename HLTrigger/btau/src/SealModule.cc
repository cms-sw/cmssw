#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/btau/interface/HLTJetTag.h"
#include "HLTrigger/btau/interface/HLTTauL25DoubleFilter.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HLTJetTag);
DEFINE_ANOTHER_FWK_MODULE(HLTTauL25DoubleFilter);
