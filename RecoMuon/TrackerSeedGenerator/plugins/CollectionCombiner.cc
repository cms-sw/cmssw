#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/TrackerSeedGenerator/plugins/CollectionCombiner.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

typedef CollectionCombiner<std::vector<Trajectory> > TrajectoryCombiner;
typedef CollectionCombiner<L3MuonTrajectorySeedCollection> L3MuonTrajectorySeedCombiner;
typedef CollectionCombiner<reco::TrackCollection> L3TrackCombiner;
typedef CollectionCombiner<TrackCandidateCollection> L3TrackCandCombiner;
typedef CollectionCombiner<reco::MuonTrackLinksCollection> L3TrackLinksCombiner;

DEFINE_FWK_MODULE(TrajectoryCombiner);
DEFINE_FWK_MODULE(L3MuonTrajectorySeedCombiner);
DEFINE_FWK_MODULE(L3TrackCombiner);
DEFINE_FWK_MODULE(L3TrackCandCombiner);
DEFINE_FWK_MODULE(L3TrackLinksCombiner);