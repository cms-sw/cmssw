#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/JetMET/interface/HLTJetVBFFilter.h"
#include "HLTrigger/JetMET/interface/HLTNVFilter.h"
#include "HLTrigger/JetMET/interface/HLTPhi2METFilter.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HLTJetVBFFilter);
DEFINE_ANOTHER_FWK_MODULE(HLTNVFilter);
DEFINE_ANOTHER_FWK_MODULE(HLTPhi2METFilter);
