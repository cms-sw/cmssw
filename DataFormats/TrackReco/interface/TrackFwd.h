#ifndef TrackReco_TrackFwd_h
#define TrackReco_TrackFwd_h
#include <vector>
#include "FWCore/EDProduct/interface/Ref.h"
#include "FWCore/EDProduct/interface/RefProd.h"
#include "FWCore/EDProduct/interface/RefVector.h"

namespace reco {
  class Track;
  typedef std::vector<Track> TrackCollection;
  typedef edm::Ref<TrackCollection> TrackRef;
  typedef edm::RefProd<TrackCollection> TracksRef;
  typedef edm::RefVector<TrackCollection> TrackRefs;
  typedef TrackRefs::iterator track_iterator;
}

#endif
