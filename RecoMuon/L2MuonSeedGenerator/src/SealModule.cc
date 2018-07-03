#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/L2MuonSeedGenerator/src/L2MuonSeedGenerator.h"
#include "RecoMuon/L2MuonSeedGenerator/src/L2MuonSeedGeneratorFromL1T.h"


DEFINE_FWK_MODULE(L2MuonSeedGenerator);
DEFINE_FWK_MODULE(L2MuonSeedGeneratorFromL1T);
