#ifndef RecoCandidate_RecoEcalCandidateFwd_h
#define RecoCandidate_RecoEcalCandidateFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class RecoEcalCandidate;

  /// collectin of RecoEcalCandidate objects
  typedef std::vector<RecoEcalCandidate> RecoEcalCandidateCollection;

  /// reference to an object in a collection of RecoEcalCandidate objects
  typedef edm::Ref<RecoEcalCandidateCollection> RecoEcalCandidateRef;

  /// reference to a collection of RecoEcalCandidate objects
  typedef edm::RefProd<RecoEcalCandidateCollection> RecoEcalCandidateRefProd;

  /// vector of objects in the same collection of RecoEcalCandidate objects
  typedef edm::RefVector<RecoEcalCandidateCollection> RecoEcalCandidateRefVector;

  /// iterator over a vector of reference to RecoEcalCandidate objects
  typedef RecoEcalCandidateRefVector::iterator recoEcalCandidate_iterator;
}  // namespace reco

#endif
