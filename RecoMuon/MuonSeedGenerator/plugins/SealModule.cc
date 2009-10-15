#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/MuonSeedGenerator/plugins/CosmicMuonSeedGenerator.h"
#include "RecoMuon/MuonSeedGenerator/plugins/MuonSeedGenerator.h"
#include "RecoMuon/MuonSeedGenerator/plugins/MuonSeedProducer.h"
#include "RecoMuon/MuonSeedGenerator/plugins/MuonSeedMerger.h"
#include "RecoMuon/MuonSeedGenerator/plugins/SETMuonSeedProducer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CosmicMuonSeedGenerator);
DEFINE_ANOTHER_FWK_MODULE(MuonSeedGenerator);
DEFINE_ANOTHER_FWK_MODULE(MuonSeedProducer);
DEFINE_ANOTHER_FWK_MODULE(MuonSeedMerger);
DEFINE_ANOTHER_FWK_MODULE(SETMuonSeedProducer);

