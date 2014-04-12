#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/Muon/interface/HLTMuonL1Filter.h"
#include "HLTrigger/Muon/interface/HLTMuonL1RegionalFilter.h"
#include "HLTrigger/Muon/interface/HLTMuonL2PreFilter.h"
#include "HLTrigger/Muon/interface/HLTMuonL3PreFilter.h"
#include "HLTrigger/Muon/interface/HLTMuonL1toL3TkPreFilter.h"
#include "HLTrigger/Muon/interface/HLTMuonIsoFilter.h"
#include "HLTrigger/Muon/interface/HLTMuonDimuonL2Filter.h"
#include "HLTrigger/Muon/interface/HLTMuonDimuonL3Filter.h"
#include "HLTrigger/Muon/interface/HLTMuonTrimuonL3Filter.h"
#include "HLTrigger/Muon/interface/HLTDiMuonGlbTrkFilter.h"
#include "HLTrigger/Muon/interface/HLTMuonPFIsoFilter.h"
DEFINE_FWK_MODULE(HLTMuonL1Filter);
DEFINE_FWK_MODULE(HLTMuonL1RegionalFilter);
DEFINE_FWK_MODULE(HLTMuonL2PreFilter);
DEFINE_FWK_MODULE(HLTMuonL3PreFilter);
DEFINE_FWK_MODULE(HLTMuonL1toL3TkPreFilter);
DEFINE_FWK_MODULE(HLTMuonIsoFilter);
DEFINE_FWK_MODULE(HLTMuonDimuonL2Filter);
DEFINE_FWK_MODULE(HLTMuonDimuonL3Filter);
DEFINE_FWK_MODULE(HLTMuonTrimuonL3Filter);
DEFINE_FWK_MODULE(HLTDiMuonGlbTrkFilter);
DEFINE_FWK_MODULE(HLTMuonPFIsoFilter);

