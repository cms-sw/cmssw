#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "RecoLocalCalo/CaloTowersCreator/src/CaloTowerCandidateCreator.h"
#include "RecoLocalCalo/CaloTowersCreator/src/CaloTowersCreator.h"
#include "RecoLocalCalo/CaloTowersCreator/src/CaloTowersReCreator.h"

DEFINE_FWK_MODULE(CaloTowersCreator);
DEFINE_FWK_MODULE(CaloTowersReCreator);
// remove following line after Jet/Met move to using
// exclusively CaloTowers
DEFINE_FWK_MODULE(CaloTowerCandidateCreator);
