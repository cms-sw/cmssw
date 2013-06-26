#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/special/interface/HLTPixlMBFilt.h"
#include "HLTrigger/special/interface/HLTPixlMBForAlignmentFilter.h"
#include "HLTrigger/special/interface/HLTPixelIsolTrackFilter.h"
#include "HLTrigger/special/interface/HLTEcalIsolationFilter.h"
#include "HLTrigger/special/interface/HLTEcalPhiSymFilter.h"
#include "HLTrigger/special/interface/HLTHcalPhiSymFilter.h"
#include "HLTrigger/special/interface/HLTHcalNoiseFilter.h"
#include "HLTrigger/special/interface/HLTHcalCalibTypeFilter.h"
#include "HLTrigger/special/interface/HLTEventNumberFilter.h"
#include "HLTrigger/special/interface/HLTL1NumberFilter.h"
#include "HLTrigger/special/interface/HLTHcalNZSFilter.h"

#include "HLTrigger/special/interface/HLTCSCOverlapFilter.h"
#include "HLTrigger/special/interface/HLTCSCRing2or3Filter.h"

#include "HLTrigger/special/interface/HLTCountNumberOfObject.h"
#include "HLTrigger/special/interface/HLTTrackWithHits.h"
#include "HLTrigger/special/interface/HLTTriggerTypeFilter.h"
#include "HLTrigger/special/interface/HLTDTROMonitorFilter.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "HLTrigger/special/interface/HLTEcalResonanceFilter.h"

#include "HLTrigger/special/interface/HLTPixelAsymmetryFilter.h"
#include "HLTrigger/special/interface/HLTHFAsymmetryFilter.h"
#include "HLTrigger/special/interface/HLTTrackerHaloFilter.h"

DEFINE_FWK_MODULE(HLTPixlMBFilt);
DEFINE_FWK_MODULE(HLTPixlMBForAlignmentFilter);
DEFINE_FWK_MODULE(HLTPixelIsolTrackFilter);
DEFINE_FWK_MODULE(HLTEcalIsolationFilter);
DEFINE_FWK_MODULE(HLTEcalPhiSymFilter);
DEFINE_FWK_MODULE(HLTHcalPhiSymFilter);
DEFINE_FWK_MODULE(HLTHcalNoiseFilter);
DEFINE_FWK_MODULE(HLTHcalCalibTypeFilter);
DEFINE_FWK_MODULE(HLTEventNumberFilter);
DEFINE_FWK_MODULE(HLTL1NumberFilter);
DEFINE_FWK_MODULE(HLTHcalNZSFilter);


DEFINE_FWK_MODULE(HLTCSCOverlapFilter);
DEFINE_FWK_MODULE(HLTCSCRing2or3Filter);

typedef HLTCountNumberOfObject<SiStripRecHit2DCollection> HLTCountNumberOfSingleRecHit;
DEFINE_FWK_MODULE(HLTCountNumberOfSingleRecHit);
typedef HLTCountNumberOfObject<SiStripMatchedRecHit2DCollection> HLTCountNumberOfMatchedRecHit;
DEFINE_FWK_MODULE(HLTCountNumberOfMatchedRecHit);
typedef HLTCountNumberOfObject<edm::View<TrajectorySeed> > HLTCountNumberOfTrajectorySeed;
DEFINE_FWK_MODULE(HLTCountNumberOfTrajectorySeed);
typedef HLTCountNumberOfObject<edm::View<reco::Track> > HLTCountNumberOfTrack;
DEFINE_FWK_MODULE(HLTCountNumberOfTrack);
DEFINE_FWK_MODULE(HLTTrackWithHits);
DEFINE_FWK_MODULE(HLTTriggerTypeFilter);

DEFINE_FWK_MODULE(HLTDTROMonitorFilter);
DEFINE_FWK_MODULE(HLTEcalResonanceFilter);

DEFINE_FWK_MODULE(HLTPixelAsymmetryFilter);
DEFINE_FWK_MODULE(HLTHFAsymmetryFilter);
DEFINE_FWK_MODULE(HLTTrackerHaloFilter);
