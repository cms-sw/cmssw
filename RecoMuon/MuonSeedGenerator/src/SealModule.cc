#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/MuonSeedGenerator/src/CosmicMuonSeedGenerator.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedGenerator.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CosmicMuonSeedGenerator);
DEFINE_ANOTHER_FWK_MODULE(MuonSeedGenerator);
