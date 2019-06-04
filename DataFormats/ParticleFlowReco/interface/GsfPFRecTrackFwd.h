#ifndef DataFormats_ParticleFlowReco_GsfPFRecTrackFwd_h
#define DataFormats_ParticleFlowReco_GsfPFRecTrackFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class GsfPFRecTrack;

  /// collection of GsfPFRecTrack objects
  typedef std::vector<GsfPFRecTrack> GsfPFRecTrackCollection;

  /// persistent reference to GsfPFRecTrack objects
  typedef edm::Ref<GsfPFRecTrackCollection> GsfPFRecTrackRef;

  /// reference to GsfPFRecTrack collection
  typedef edm::RefProd<GsfPFRecTrackCollection> GsfPFRecTrackRefProd;

  /// vector of references to GsfPFRecTrack objects all in the same collection
  typedef edm::RefVector<GsfPFRecTrackCollection> GsfPFRecTrackRefVector;

  /// iterator over a vector of references to GsfPFRecTrack objects
  typedef GsfPFRecTrackRefVector::iterator gsfPfRecTrack_iterator;
}  // namespace reco

#endif
