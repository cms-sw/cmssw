#ifndef HcalIsolatedTrack_HcalIsolatedTrackCandidateFwd_h
#define HcalIsolatedTrack_HcalIsolatedTrackCandidateFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class HcalIsolatedTrackCandidate;

  /// collectin of HcalIsolatedTrackCandidate objects
  typedef std::vector<HcalIsolatedTrackCandidate> HcalIsolatedTrackCandidateCollection;

  /// reference to an object in a collection of HcalIsolatedTrackCandidate objects
  typedef edm::Ref<HcalIsolatedTrackCandidateCollection> HcalIsolatedTrackCandidateRef;

  /// reference to a collection of HcalIsolatedTrackCandidate objects
  typedef edm::RefProd<HcalIsolatedTrackCandidateCollection> HcalIsolatedTrackCandidateRefProd;

  /// vector of objects in the same collection of HcalIsolatedTrackCandidate objects
  typedef edm::RefVector<HcalIsolatedTrackCandidateCollection> HcalIsolatedTrackCandidateRefVector;

  /// iterator over a vector of reference to HcalIsolatedTrackCandidate objects
  typedef HcalIsolatedTrackCandidateRefVector::iterator HcalIsolatedTrackCandidateIterator;

  typedef std::vector<reco::HcalIsolatedTrackCandidateRef> HcalIsolatedTrackCandidateSimpleRefVector;
}

#endif

