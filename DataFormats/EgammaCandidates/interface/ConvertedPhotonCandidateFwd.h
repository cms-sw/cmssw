#ifndef EgammaReco_ConvertedPhotonCandidateFwd_h
#define EgammaReco_ConvertedPhotonCandidateFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class ConvertedPhotonCandidate;

  /// collectin of ConvertedPhotonCandidate objects
  typedef std::vector<ConvertedPhotonCandidate> ConvertedPhotonCandidateCollection;

  /// reference to an object in a collection of ConvertedPhotonCandidate objects
  typedef edm::Ref<ConvertedPhotonCandidateCollection> ConvertedPhotonCandidateRef;

  /// reference to a collection of ConvertedPhotonCandidate objects
  typedef edm::RefProd<ConvertedPhotonCandidateCollection> ConvertedPhotonCandidateRefProd;

  /// vector of objects in the same collection of ConvertedPhotonCandidate objects
  typedef edm::RefVector<ConvertedPhotonCandidateCollection> ConvertedPhotonCandidateRefVector;

  /// iterator over a vector of reference to ConvertedPhotonCandidate objects
  typedef ConvertedPhotonCandidateRefVector::iterator convPhotonCandidate_iterator;
}

#endif
