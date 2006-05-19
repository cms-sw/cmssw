#ifndef EgammaReco_PhotonCandidateFwd_h
#define EgammaReco_PhotonCandidateFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class PhotonCandidate;

  /// collectin of PhotonCandidate objects
  typedef std::vector<PhotonCandidate> PhotonCandidateCollection;

  /// reference to an object in a collection of PhotonCandidate objects
  typedef edm::Ref<PhotonCandidateCollection> PhotonCandidateRef;

  /// reference to a collection of PhotonCandidate objects
  typedef edm::RefProd<PhotonCandidateCollection> PhotonCandidateRefProd;

  /// vector of objects in the same collection of PhotonCandidate objects
  typedef edm::RefVector<PhotonCandidateCollection> PhotonCandidateRefVector;

  /// iterator over a vector of reference to PhotonCandidate objects
  typedef PhotonCandidateRefVector::iterator photonCandidate_iterator;
}

#endif
