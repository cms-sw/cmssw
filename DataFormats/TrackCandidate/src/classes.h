#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/CLHEP/interface/Migration.h" 
#include <boost/cstdint.hpp> 
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h" 
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace DataFormats_TrackCandidate {
  struct dictionary {
    TrackCandidate tc;
    TrackCandidateCollection coll;
    edm::Wrapper<TrackCandidateCollection> TrackCandidateCollectionWrapper;
  };
}

