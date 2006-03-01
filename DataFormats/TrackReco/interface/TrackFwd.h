#ifndef TrackReco_TrackFwd_h
#define TrackReco_TrackFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class Track;
  /// collection of Tracks
  typedef std::vector<Track> TrackCollection;
  /// persistent reference to a Track
  typedef edm::Ref<TrackCollection> TrackRef;
  /// persistent reference to a Track collection
  typedef edm::RefProd<TrackCollection> TracksRef;
  /// vector of reference to Track in the same collection
  typedef edm::RefVector<TrackCollection> TrackRefs;
  /// iterator over a vector of reference to Track in the same collection
  typedef TrackRefs::iterator track_iterator;
}

#endif
