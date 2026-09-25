#include "DQMServices/Components/interface/GenericObjectDQMSource.h"
#include "DQMOffline/Trigger/interface/TrackDQMVariables.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using TrackGenericDQMSource = GenericObjectDQMSource<reco::Track>;

DEFINE_FWK_MODULE(TrackGenericDQMSource);
