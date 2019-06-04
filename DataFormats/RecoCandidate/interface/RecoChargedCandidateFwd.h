#ifndef RecoCandidate_RecoChargedCandidateFwd_h
#define RecoCandidate_RecoChargedCandidateFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class RecoChargedCandidate;

  /// collectin of RecoChargedCandidate objects
  typedef std::vector<RecoChargedCandidate> RecoChargedCandidateCollection;

  /// reference to an object in a collection of RecoChargedCandidate objects
  typedef edm::Ref<RecoChargedCandidateCollection> RecoChargedCandidateRef;

  /// reference to a collection of RecoChargedCandidate objects
  typedef edm::RefProd<RecoChargedCandidateCollection> RecoChargedCandidateRefProd;

  /// vector of objects in the same collection of RecoChargedCandidate objects
  typedef edm::RefVector<RecoChargedCandidateCollection> RecoChargedCandidateRefVector;

  /// iterator over a vector of reference to RecoChargedCandidate objects
  typedef RecoChargedCandidateRefVector::iterator recoChargedCandidate_iterator;
}  // namespace reco

#endif
