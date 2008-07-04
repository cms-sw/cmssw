#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMOffline/JetMET/src/JetMETAnalyzer.h"

// the clients
// #include "DQMOffline/Muon/src/MuonTrackResidualsTest.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(JetMETAnalyzer);
// DEFINE_ANOTHER_FWK_MODULE(MuonTrackResidualsTest);
