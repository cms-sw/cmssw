#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/CosmicMuonProducer/src/CosmicMuonProducer.h"
#include "RecoMuon/CosmicMuonProducer/src/GlobalCosmicMuonProducer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CosmicMuonProducer);
DEFINE_ANOTHER_FWK_MODULE(GlobalCosmicMuonProducer);
