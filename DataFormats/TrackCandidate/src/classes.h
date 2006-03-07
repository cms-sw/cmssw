#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

namespace {
  namespace {
    TrackCandidate tc;
    TrackCandidateCollection coll;
    edm::Wrapper<TrackCandidateCollection> TrackCandidateCollectionWrapper;
  }
}

