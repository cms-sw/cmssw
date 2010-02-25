#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/MuonSeedGenerator/plugins/CosmicMuonSeedGenerator.h"
#include "RecoMuon/MuonSeedGenerator/plugins/MuonSeedGenerator.h"
#include "RecoMuon/MuonSeedGenerator/plugins/MuonSeedProducer.h"
#include "RecoMuon/MuonSeedGenerator/plugins/MuonSeedMerger.h"
#include "RecoMuon/MuonSeedGenerator/plugins/SETMuonSeedProducer.h"


DEFINE_FWK_MODULE(CosmicMuonSeedGenerator);
DEFINE_FWK_MODULE(MuonSeedGenerator);
DEFINE_FWK_MODULE(MuonSeedProducer);
DEFINE_FWK_MODULE(MuonSeedMerger);
DEFINE_FWK_MODULE(SETMuonSeedProducer);

