#ifndef TrackReco_TrackFwd_h
#define TrackReco_TrackFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class Track;
  typedef std::vector<Track> TrackCollection;
  typedef edm::Ref<TrackCollection> TrackRef;
  typedef edm::RefProd<TrackCollection> TracksRef;
  typedef edm::RefVector<TrackCollection> TrackRefs;
  typedef TrackRefs::iterator track_iterator;
}

#endif
