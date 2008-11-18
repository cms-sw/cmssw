#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/MuonSeedGenerator/src/CosmicMuonSeedGenerator.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedGenerator.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedProducer.h"
#include "RecoMuon/MuonSeedGenerator/src/RPCSeedGenerator.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CosmicMuonSeedGenerator);
DEFINE_ANOTHER_FWK_MODULE(MuonSeedGenerator);
DEFINE_ANOTHER_FWK_MODULE(MuonSeedProducer);
DEFINE_ANOTHER_FWK_MODULE(RPCSeedGenerator);

