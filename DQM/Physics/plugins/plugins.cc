#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQM/Physics/src/BPhysicsOniaDQM.h"
#include "DQM/Physics/src/EwkDQM.h"
#include "DQM/Physics/src/EwkMuDQM.h"
#include "DQM/Physics/src/EwkMuLumiMonitorDQM.h"
#include "DQM/Physics/src/EwkElecDQM.h"
#include "DQM/Physics/src/EwkTauDQM.h"
#include "DQM/Physics/src/QcdPhotonsDQM.h"
#include "DQM/Physics/src/QcdLowPtDQM.h"
#include "DQM/Physics/src/QcdHighPtDQM.h"
#include "DQM/Physics/src/TopDiLeptonDQM.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(BPhysicsOniaDQM);
DEFINE_ANOTHER_FWK_MODULE(EwkDQM);
DEFINE_ANOTHER_FWK_MODULE(EwkMuDQM);
DEFINE_ANOTHER_FWK_MODULE(EwkMuLumiMonitorDQM);
DEFINE_ANOTHER_FWK_MODULE(EwkElecDQM);
DEFINE_ANOTHER_FWK_MODULE(EwkTauDQM);
DEFINE_ANOTHER_FWK_MODULE(QcdPhotonsDQM);
DEFINE_ANOTHER_FWK_MODULE(QcdLowPtDQM);
DEFINE_ANOTHER_FWK_MODULE(QcdHighPtDQM);
DEFINE_ANOTHER_FWK_MODULE(TopDiLeptonDQM);
