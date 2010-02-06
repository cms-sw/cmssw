#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMOffline/JetMET/interface/JetMETAnalyzer.h"
#include "DQMOffline/JetMET/interface/CaloTowerAnalyzer.h"
#include "DQMOffline/JetMET/interface/ECALRecHitAnalyzer.h"
#include "DQMOffline/JetMET/interface/HCALRecHitAnalyzer.h"
#include "DQMOffline/JetMET/interface/DataCertificationJetMET.h"
#include "DQMOffline/JetMET/interface/JetMETDQMOfflineClient.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(JetMETAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CaloTowerAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(HCALRecHitAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(ECALRecHitAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(DataCertificationJetMET);
DEFINE_ANOTHER_FWK_MODULE(JetMETDQMOfflineClient);

