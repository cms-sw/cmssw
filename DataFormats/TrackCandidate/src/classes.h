#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/CLHEP/interface/Migration.h" 
#include <boost/cstdint.hpp> 
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h" 
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateSeedAssociation.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  struct dictionary {
    TrackCandidate tc;
    TrackCandidateCollection coll;
    edm::Wrapper<TrackCandidateCollection> TrackCandidateCollectionWrapper;

    reco::TrackCandidateSeedAssociationCollection v5;
    edm::Wrapper<reco::TrackCandidateSeedAssociationCollection> c5;
    reco::TrackCandidateSeedAssociation vv5;
    reco::TrackCandidateSeedAssociationRef r5;
    reco::TrackCandidateSeedAssociationRefProd rp5;
    reco::TrackCandidateSeedAssociationRefVector rv5;

  };
}

