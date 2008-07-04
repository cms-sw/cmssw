#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMOffline/Muon/src/MuonAnalyzer.h"

// the clients
#include "DQMOffline/Muon/src/MuonTrackResidualsTest.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MuonAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(MuonTrackResidualsTest);
