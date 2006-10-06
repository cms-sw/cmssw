#ifndef TrackReco_GsfTrackExtraFwd_h
#define TrackReco_GsfTrackExtraFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class GsfTrackExtra;
  /// collection of GsfTrackExtra objects
  typedef std::vector<GsfTrackExtra> GsfTrackExtraCollection;
  /// persistent reference to a GsfTrackExtra
  typedef edm::Ref<GsfTrackExtraCollection> GsfTrackExtraRef;
  /// reference to a GsfTrackExtra collection
  typedef edm::RefProd<GsfTrackExtraCollection> GsfTrackExtraRefProd;
  /// vector of references to GsfTrackExtra in the same collection
  typedef edm::RefVector<GsfTrackExtraCollection> GsfTrackExtraRefVector;
  /// iterator over a vector of references to GsfTrackExtra in the same collection
  typedef GsfTrackExtraRefVector::iterator gsfTrackExtra_iterator;
}

#endif
