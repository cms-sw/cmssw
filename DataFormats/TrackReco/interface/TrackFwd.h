#ifndef TrackReco_TrackFwd_h
#define TrackReco_TrackFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class Track;
  /// a collection of Tracks
  typedef std::vector<Track> TrackCollection;
  /// a persistent reference to a Track
  typedef edm::Ref<TrackCollection> TrackRef;
  /// a persistent reference to a Track collection
  typedef edm::RefProd<TrackCollection> TracksRef;
  /// a  vector of reference to Track in the same collection
  typedef edm::RefVector<TrackCollection> TrackRefs;
  /// iterator over a vector of reference to Track in the same collection
  typedef TrackRefs::iterator track_iterator;
}

#endif
