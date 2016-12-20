#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMOffline/JetMET/interface/JetAnalyzer.h"
#include "DQMOffline/JetMET/interface/JetAnalyzer_HeavyIons.h"
#include "DQMOffline/JetMET/interface/JetAnalyzer_HeavyIons_matching.h"
#include "DQMOffline/JetMET/interface/METAnalyzer.h"
#include "DQMOffline/JetMET/interface/CaloTowerAnalyzer.h"
#include "DQMOffline/JetMET/interface/ECALRecHitAnalyzer.h"
#include "DQMOffline/JetMET/interface/HCALRecHitAnalyzer.h"
#include "DQMOffline/JetMET/interface/BeamHaloAnalyzer.h"
#include "DQMOffline/JetMET/interface/DataCertificationJetMET.h"
#include "DQMOffline/JetMET/interface/JetMETDQMDCSFilter.h"
#include "DQMOffline/JetMET/interface/SUSYDQMAnalyzer.h"

DEFINE_FWK_MODULE(JetAnalyzer);
DEFINE_FWK_MODULE(JetAnalyzer_HeavyIons);
DEFINE_FWK_MODULE(JetAnalyzer_HeavyIons_matching);
DEFINE_FWK_MODULE(METAnalyzer);
DEFINE_FWK_MODULE(CaloTowerAnalyzer);
DEFINE_FWK_MODULE(HCALRecHitAnalyzer);
DEFINE_FWK_MODULE(ECALRecHitAnalyzer);
DEFINE_FWK_MODULE(BeamHaloAnalyzer);
DEFINE_FWK_MODULE(DataCertificationJetMET);
DEFINE_FWK_MODULE(SUSYDQMAnalyzer);
