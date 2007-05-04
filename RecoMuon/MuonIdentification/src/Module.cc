#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/MuonIdentification/interface/MuonIdProducer.h"
#include "RecoMuon/MuonIdentification/src/MuonProducer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MuonIdProducer);
DEFINE_ANOTHER_FWK_MODULE(MuonProducer);
