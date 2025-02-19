#ifndef GsfTrackReco_GsfTrackFwd_h
#define GsfTrackReco_GsfTrackFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class GsfTrack;
  /// collection of GsfTracks
  typedef std::vector<GsfTrack> GsfTrackCollection;
  /// persistent reference to a GsfTrack
  typedef edm::Ref<GsfTrackCollection> GsfTrackRef;
  /// persistent reference to a GsfTrack collection
  typedef edm::RefProd<GsfTrackCollection> GsfTrackRefProd;
  /// vector of reference to GsfTrack in the same collection
  typedef edm::RefVector<GsfTrackCollection> GsfTrackRefVector;
  /// iterator over a vector of reference to GsfTrack in the same collection
  typedef GsfTrackRefVector::iterator GsfTrack_iterator;
}

#endif
