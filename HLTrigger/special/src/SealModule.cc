#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/special/interface/HLTPixlMBFilt.h"
#include "HLTrigger/special/interface/HLTPixlMBForAlignmentFilter.h"
#include "HLTrigger/special/interface/HLTPixelIsolTrackFilter.h"
#include "HLTrigger/special/interface/HLTEcalIsolationFilter.h"
#include "HLTrigger/special/interface/HLTEcalPhiSymFilter.h"
#include "HLTrigger/special/interface/HLTHcalPhiSymFilter.h"
#include "HLTrigger/special/interface/HLTPi0RecHitsFilter.h"
#include "HLTrigger/special/interface/HLTCSCOverlapFilter.h"
#include "HLTrigger/special/interface/HLTCSCRing2or3Filter.h"

#include "HLTrigger/special/interface/CountNumberOfObject.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/RoadSearchSeed/interface/RoadSearchSeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"


DEFINE_FWK_MODULE(HLTPixlMBFilt);
DEFINE_FWK_MODULE(HLTPixlMBForAlignmentFilter);
DEFINE_FWK_MODULE(HLTPixelIsolTrackFilter);
DEFINE_FWK_MODULE(HLTEcalIsolationFilter);
DEFINE_FWK_MODULE(HLTEcalPhiSymFilter);
DEFINE_FWK_MODULE(HLTHcalPhiSymFilter);
DEFINE_FWK_MODULE(HLTPi0RecHitsFilter);
DEFINE_FWK_MODULE(HLTCSCOverlapFilter);
DEFINE_FWK_MODULE(HLTCSCRing2or3Filter);

typedef CountNumberOfObject<SiStripRecHit2DCollection> HLTCountNumberOfSingleRecHit;
DEFINE_FWK_MODULE(HLTCountNumberOfSingleRecHit);
typedef CountNumberOfObject<SiStripMatchedRecHit2DCollection> HLTCountNumberOfMatchedRecHit;
DEFINE_FWK_MODULE(HLTCountNumberOfMatchedRecHit);
typedef CountNumberOfObject<edm::View<TrajectorySeed> > HLTCountNumberOfTrajectorySeed;
DEFINE_FWK_MODULE(HLTCountNumberOfTrajectorySeed);
typedef CountNumberOfObject<RoadSearchSeedCollection> HLTCountNumberOfRoadSearchSeed;
DEFINE_FWK_MODULE(HLTCountNumberOfRoadSearchSeed);
typedef CountNumberOfObject<edm::View<reco::Track> > HLTCountNumberOfTrack;
DEFINE_FWK_MODULE(HLTCountNumberOfTrack);

