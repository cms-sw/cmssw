#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "DQM/Physics/src/BPhysicsOniaDQM.h"
#include "DQM/Physics/src/QcdPhotonsPhotonJetDQM.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(BPhysicsOniaDQM);
DEFINE_ANOTHER_FWK_MODULE(QcdPhotonsPhotonJetDQM);
