#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/MuonIdentification/plugins/MuonIdProducer.h"
#include "RecoMuon/MuonIdentification/plugins/MuonLinksProducer.h"
#include "RecoMuon/MuonIdentification/plugins/MuonLinksProducerForHLT.h"
#include "RecoMuon/MuonIdentification/plugins/MuonRefProducer.h"
#include "RecoMuon/MuonIdentification/plugins/MuonProducer.h"
#include "RecoMuon/MuonIdentification/plugins/MuonTimingProducer.h"
#include "RecoMuon/MuonIdentification/plugins/MuonSelectionTypeValueMapProducer.h"
#include "RecoMuon/MuonIdentification/plugins/InterestingEcalDetIdProducer.h"


DEFINE_FWK_MODULE(MuonIdProducer);
DEFINE_FWK_MODULE(MuonLinksProducer);
DEFINE_FWK_MODULE(MuonLinksProducerForHLT);
DEFINE_FWK_MODULE(MuonRefProducer);
DEFINE_FWK_MODULE(MuonProducer);
DEFINE_FWK_MODULE(MuonTimingProducer);
DEFINE_FWK_MODULE(MuonSelectionTypeValueMapProducer);
DEFINE_FWK_MODULE(InterestingEcalDetIdProducer);

// For the VID framework
#include "PhysicsTools/SelectorUtils/interface/VersionedIdProducer.h"
typedef VersionedIdProducer<reco::MuonPtr> VersionedMuonIdProducer;
DEFINE_FWK_MODULE(VersionedMuonIdProducer);
