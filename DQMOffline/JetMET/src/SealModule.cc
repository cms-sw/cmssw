#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMOffline/JetMET/src/JetMETAnalyzer.h"
#include "DQMOffline/JetMET/interface/CaloTowerAnalyzer.h"
#include "DQMOffline/JetMET/interface/ECALRecHitAnalyzer.h"
#include "DQMOffline/JetMET/interface/HCALRecHitAnalyzer.h"
// the clients
// #include "DQMOffline/Muon/src/MuonTrackResidualsTest.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(JetMETAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CaloTowerAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(HCALRecHitAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(ECALRecHitAnalyzer);
// DEFINE_ANOTHER_FWK_MODULE(MuonTrackResidualsTest);
