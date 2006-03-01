#ifndef TrackReco_TrackExtraFwd_h
#define TrackReco_TrackExtraFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class TrackExtra;
  /// collection of TrackExtra objects
  typedef std::vector<TrackExtra> TrackExtraCollection;
  /// persistent reference to a TrackExtra
  typedef edm::Ref<TrackExtraCollection> TrackExtraRef;
  /// vector of references to TrackExtra in the same collection
  typedef edm::RefVector<TrackExtraCollection> TrackExtraRefs;
  /// iterator over a vector of references to TrackExtra in the same collection
  typedef TrackExtraRefs::iterator trackExtra_iterator;
}

#endif
