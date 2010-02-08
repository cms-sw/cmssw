#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMOffline/Muon/src/MuonAnalyzer.h"
#include "DQMOffline/Muon/interface/DTSegmentsTask.h"

// the clients
#include "DQMOffline/Muon/src/MuonTrackResidualsTest.h"
#include "DQMOffline/Muon/src/MuonRecoTest.h"
#include "DQMOffline/Muon/src/MuonTestSummary.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MuonAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(MuonTrackResidualsTest);
DEFINE_ANOTHER_FWK_MODULE(MuonRecoTest);
DEFINE_ANOTHER_FWK_MODULE(DTSegmentsTask);
DEFINE_ANOTHER_FWK_MODULE(MuonTestSummary);

