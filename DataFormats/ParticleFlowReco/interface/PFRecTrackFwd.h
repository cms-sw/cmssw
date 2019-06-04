#ifndef DataFormats_ParticleFlowReco_PFRecTrackFwd_h
#define DataFormats_ParticleFlowReco_PFRecTrackFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class PFRecTrack;

  /// collection of PFRecTrack objects
  typedef std::vector<PFRecTrack> PFRecTrackCollection;

  /// persistent reference to PFRecTrack objects
  typedef edm::Ref<PFRecTrackCollection> PFRecTrackRef;

  /// reference to PFRecTrack collection
  typedef edm::RefProd<PFRecTrackCollection> PFRecTrackRefProd;

  /// vector of references to PFRecTrack objects all in the same collection
  typedef edm::RefVector<PFRecTrackCollection> PFRecTrackRefVector;

  /// iterator over a vector of references to PFRecTrack objects
  typedef PFRecTrackRefVector::iterator pfRecTrack_iterator;
}  // namespace reco

#endif
