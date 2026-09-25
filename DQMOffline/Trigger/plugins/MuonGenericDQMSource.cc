#include "DQMServices/Components/interface/GenericObjectDQMSource.h"
#include "DQMOffline/Trigger/interface/MuonDQMVariables.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using MuonGenericDQMSource = GenericObjectDQMSource<reco::Muon>;

DEFINE_FWK_MODULE(MuonGenericDQMSource);
