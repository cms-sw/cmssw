#define TRACKCANDIDATE_CLASSES_H
#ifndef TRACKCANDIDATE_CLASSES_H

#include "DataFormats/Trackcandidate/interface/TrackCandidateCollection.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

namespace {
  namespace {
    TrackCandidateCollection coll;
    edm::Wrapper<TrackCandidateCollection> TrackCandidateCollectionWrapper;
  }
}

#endif
