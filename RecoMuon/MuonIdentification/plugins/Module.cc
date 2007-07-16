#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/MuonIdentification/plugins/MuonIdProducer.h"
#include "RecoMuon/MuonIdentification/plugins/MuonRefProducer.h"
#include "RecoMuon/MuonIdentification/plugins/MuonProducer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MuonIdProducer);
DEFINE_ANOTHER_FWK_MODULE(MuonRefProducer);
DEFINE_ANOTHER_FWK_MODULE(MuonProducer);
