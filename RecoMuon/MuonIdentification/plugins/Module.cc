#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/MuonIdentification/plugins/MuonIdProducer.h"
#include "RecoMuon/MuonIdentification/plugins/MuonLinksProducer.h"
#include "RecoMuon/MuonIdentification/plugins/MuonRefProducer.h"
#include "RecoMuon/MuonIdentification/plugins/MuonProducer.h"
#include "RecoMuon/MuonIdentification/plugins/MuonTimingProducer.h"
#include "RecoMuon/MuonIdentification/plugins/MuonSelectionTypeValueMapProducer.h"
#include "RecoMuon/MuonIdentification/plugins/InterestingEcalDetIdProducer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MuonIdProducer);
DEFINE_ANOTHER_FWK_MODULE(MuonLinksProducer);
DEFINE_ANOTHER_FWK_MODULE(MuonRefProducer);
DEFINE_ANOTHER_FWK_MODULE(MuonProducer);
DEFINE_ANOTHER_FWK_MODULE(MuonTimingProducer);
DEFINE_ANOTHER_FWK_MODULE(MuonSelectionTypeValueMapProducer);
DEFINE_ANOTHER_FWK_MODULE(InterestingEcalDetIdProducer);
