#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/CaloTowersCreator/src/CaloTowersCreator.h"
#include "RecoLocalCalo/CaloTowersCreator/src/CaloTowersReCreator.h"
#include "RecoLocalCalo/CaloTowersCreator/src/CaloTowerCandidateCreator.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( CaloTowersCreator );
DEFINE_ANOTHER_FWK_MODULE( CaloTowersReCreator );
