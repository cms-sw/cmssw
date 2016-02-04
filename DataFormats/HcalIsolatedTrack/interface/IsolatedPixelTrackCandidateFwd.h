#ifndef HcalIsolatedTrack_IsolatedPixelTrackCandidateFwd_h
#define HcalIsolatedTrack_IsolatedPixelTrackCandidateFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class IsolatedPixelTrackCandidate;

  /// collectin of IsolatedPixelTrackCandidate objects
  typedef std::vector<IsolatedPixelTrackCandidate> IsolatedPixelTrackCandidateCollection;

  /// reference to an object in a collection of IsolatedPixelTrackCandidate objects
  typedef edm::Ref<IsolatedPixelTrackCandidateCollection> IsolatedPixelTrackCandidateRef;

  /// reference to a collection of IsolatedPixelTrackCandidate objects
  typedef edm::RefProd<IsolatedPixelTrackCandidateCollection> IsolatedPixelTrackCandidateRefProd;

  /// vector of objects in the same collection of IsolatedPixelTrackCandidate objects
  typedef edm::RefVector<IsolatedPixelTrackCandidateCollection> IsolatedPixelTrackCandidateRefVector;

  /// iterator over a vector of reference to IsolatedPixelTrackCandidate objects
  typedef IsolatedPixelTrackCandidateRefVector::iterator IsolatedPixelTrackCandidateIterator;

  typedef std::vector<reco::IsolatedPixelTrackCandidateRef> IsolatedPixelTrackCandidateSimpleRefVector;
}

#endif

