#ifndef Candidate_NamedCompositeCandidateFwd_h
#define Candidate_NamedCompositeCandidateFwd_h
#include "DataFormats/Common/interface/OwnVector.h"

namespace reco {
  class NamedCompositeCandidate;
}

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include "DataFormats/Common/interface/View.h"

namespace reco {
  /// collection of Candidate objects
  typedef std::vector<NamedCompositeCandidate> NamedCompositeCandidateCollection;
  /// view of a collection containing candidates
  typedef edm::View<NamedCompositeCandidate> NamedCompositeCandidateView;
  /// persistent reference to an object in a collection of Candidate objects
  typedef edm::Ref<NamedCompositeCandidateCollection> NamedCompositeCandidateRef;
  /// persistent reference to an object in a collection of Candidate objects
  typedef edm::RefToBase<NamedCompositeCandidate> NamedCompositeCandidateBaseRef;
  /// vector of references to objects in the same  collection of Candidate objects
  typedef edm::RefVector<NamedCompositeCandidateCollection> NamedCompositeCandidateRefVector;
  //  /// vector of references to objects in the same collection of Candidate objects via base type
  //  typedef edm::RefToBaseVector<NamedCompositeCandidate> NamedCompositeCandidateBaseRefVector;
  /// reference to a collection of Candidate objects
  typedef edm::RefProd<NamedCompositeCandidateCollection> NamedCompositeCandidateRefProd;
  /// vector of references to objects in the same collection of Candidate objects via base type
  typedef edm::RefToBaseProd<NamedCompositeCandidate> NamedCompositeCandidateBaseRefProd;
}  // namespace reco

#endif
