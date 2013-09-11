#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMOffline/Muon/interface/MuonAnalyzer.h"
#include "DQMOffline/Muon/interface/DTSegmentsTask.h"

// the clients
#include "DQMOffline/Muon/interface/MuonTrackResidualsTest.h"
#include "DQMOffline/Muon/interface/MuonRecoTest.h"
#include "DQMOffline/Muon/interface/EfficiencyPlotter.h"
#include "DQMOffline/Muon/interface/MuonTestSummary.h"


DEFINE_FWK_MODULE(MuonAnalyzer);
DEFINE_FWK_MODULE(MuonTrackResidualsTest);
DEFINE_FWK_MODULE(MuonRecoTest);
DEFINE_FWK_MODULE(EfficiencyPlotter);
DEFINE_FWK_MODULE(DTSegmentsTask);
DEFINE_FWK_MODULE(MuonTestSummary);

