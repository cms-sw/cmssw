#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "HLTrigger/special/plugins/HLTCountNumberOfObject.h"

// declare these template instantiations as framework plugins
#include "FWCore/Framework/interface/MakerMacros.h"

using HLTCountNumberOfSingleRecHit = HLTCountNumberOfObject<SiStripRecHit2DCollection>;
DEFINE_FWK_MODULE(HLTCountNumberOfSingleRecHit);

using HLTCountNumberOfMatchedRecHit = HLTCountNumberOfObject<SiStripMatchedRecHit2DCollection>;
DEFINE_FWK_MODULE(HLTCountNumberOfMatchedRecHit);

using HLTCountNumberOfTrajectorySeed = HLTCountNumberOfObject<edm::View<TrajectorySeed>>;
DEFINE_FWK_MODULE(HLTCountNumberOfTrajectorySeed);

using HLTCountNumberOfTrack = HLTCountNumberOfObject<edm::View<reco::Track>>;
DEFINE_FWK_MODULE(HLTCountNumberOfTrack);
