#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/Muon/interface/HLTMuonL1Filter.h"
#include "HLTrigger/Muon/interface/HLTMuonPreFilter.h"
#include "HLTrigger/Muon/interface/HLTMuonL2PreFilter.h"
#include "HLTrigger/Muon/interface/HLTMuonL3PreFilter.h"
#include "HLTrigger/Muon/interface/HLTMuonL3TkPreFilter.h"
#include "HLTrigger/Muon/interface/HLTMuonIsoFilter.h"
#include "HLTrigger/Muon/interface/HLTMuonDimuonFilter.h"
#include "HLTrigger/Muon/interface/HLTMuonDimuonL2Filter.h"
#include "HLTrigger/Muon/interface/HLTMuonDimuonL3Filter.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HLTMuonL1Filter);
DEFINE_ANOTHER_FWK_MODULE(HLTMuonL2PreFilter);
DEFINE_ANOTHER_FWK_MODULE(HLTMuonL3PreFilter);
DEFINE_ANOTHER_FWK_MODULE(HLTMuonL3TkPreFilter);
DEFINE_ANOTHER_FWK_MODULE(HLTMuonPreFilter);
DEFINE_ANOTHER_FWK_MODULE(HLTMuonIsoFilter);
DEFINE_ANOTHER_FWK_MODULE(HLTMuonDimuonFilter);
DEFINE_ANOTHER_FWK_MODULE(HLTMuonDimuonL2Filter);
DEFINE_ANOTHER_FWK_MODULE(HLTMuonDimuonL3Filter);
