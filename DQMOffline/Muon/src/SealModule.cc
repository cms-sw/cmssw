#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMOffline/Muon/interface/DTSegmentsTask.h"
#include "DQMOffline/Muon/interface/MuonTrackResidualsTest.h"
#include "DQMOffline/Muon/interface/MuonRecoTest.h"
#include "DQMOffline/Muon/interface/EfficiencyPlotter.h"
#include "DQMOffline/Muon/interface/MuonTestSummary.h"

#include "DQMOffline/Muon/interface/DiMuonHistograms.h"
#include "DQMOffline/Muon/interface/MuonKinVsEtaAnalyzer.h"
#include "DQMOffline/Muon/interface/EfficiencyAnalyzer.h"
#include "DQMOffline/Muon/interface/MuonRecoOneHLT.h"
#include "DQMOffline/Muon/interface/MuonRecoAnalyzer.h"
#include "DQMOffline/Muon/interface/MuonTiming.h"
#include "DQMOffline/Muon/interface/SegmentTrackAnalyzer.h"
#include "DQMOffline/Muon/interface/MuonSeedsAnalyzer.h"
#include "DQMOffline/Muon/interface/MuonEnergyDepositAnalyzer.h"

#include "DQMOffline/Muon/interface/MuonMiniAOD.h"

#include "DQMOffline/Muon/interface/TriggerMatchMonitor.h"
#include "DQMOffline/Muon/interface/TriggerMatchEfficiencyPlotter.h"
#include "DQMOffline/Muon/interface/GEMOfflineMonitor.h"
#include "DQMOffline/Muon/interface/GEMEfficiencyAnalyzer.h"
#include "DQMOffline/Muon/interface/GEMEfficiencyHarvester.h"

DEFINE_FWK_MODULE(MuonTrackResidualsTest);
DEFINE_FWK_MODULE(MuonRecoTest);
DEFINE_FWK_MODULE(EfficiencyPlotter);
DEFINE_FWK_MODULE(DTSegmentsTask);
DEFINE_FWK_MODULE(MuonTestSummary);
DEFINE_FWK_MODULE(DiMuonHistograms);
DEFINE_FWK_MODULE(MuonKinVsEtaAnalyzer);
DEFINE_FWK_MODULE(EfficiencyAnalyzer);
DEFINE_FWK_MODULE(MuonRecoOneHLT);
DEFINE_FWK_MODULE(MuonRecoAnalyzer);
DEFINE_FWK_MODULE(MuonTiming);
DEFINE_FWK_MODULE(SegmentTrackAnalyzer);
DEFINE_FWK_MODULE(MuonEnergyDepositAnalyzer);
DEFINE_FWK_MODULE(MuonSeedsAnalyzer);
DEFINE_FWK_MODULE(MuonMiniAOD);
DEFINE_FWK_MODULE(TriggerMatchMonitor);
DEFINE_FWK_MODULE(TriggerMatchEfficiencyPlotter);
DEFINE_FWK_MODULE(GEMOfflineMonitor);
DEFINE_FWK_MODULE(GEMEfficiencyAnalyzer);
DEFINE_FWK_MODULE(GEMEfficiencyHarvester);
