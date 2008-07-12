#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMOffline/Muon/src/MuonAnalyzer.h"
#include "DQMOffline/Muon/interface/RPCEfficiency.h"
#include "DQMOffline/Muon/interface/RPCEfficiencySecond.h"

// the clients
#include "DQMOffline/Muon/src/MuonTrackResidualsTest.h"
#include "DQMOffline/Muon/src/MuonRecoTest.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MuonAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(MuonTrackResidualsTest);
DEFINE_ANOTHER_FWK_MODULE(MuonRecoTest);
DEFINE_ANOTHER_FWK_MODULE(RPCEfficiency);
DEFINE_ANOTHER_FWK_MODULE(RPCEfficiencySecond);

