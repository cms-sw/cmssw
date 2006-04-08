#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/CaloTowersCreator/interface/CaloTowersCreator.h"
#include "RecoLocalCalo/CaloTowersCreator/interface/CaloTowerCandidateCreator.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( CaloTowersCreator );
DEFINE_ANOTHER_FWK_MODULE( CaloTowerCandidateCreator );
