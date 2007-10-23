#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/Muon/interface/HLTMuonL1Filter.h"
#include "HLTrigger/Muon/interface/HLTMuonPreFilter.h"
#include "HLTrigger/Muon/interface/HLTMuonIsoFilter.h"
#include "HLTrigger/Muon/interface/HLTMuonDimuonFilter.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HLTMuonL1Filter);
DEFINE_ANOTHER_FWK_MODULE(HLTMuonPreFilter);
DEFINE_ANOTHER_FWK_MODULE(HLTMuonIsoFilter);
DEFINE_ANOTHER_FWK_MODULE(HLTMuonDimuonFilter);
