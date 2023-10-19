#ifndef Candidate_CompositeCandidateFwd_h
#define Candidate_CompositeCandidateFwd_h
#include "DataFormats/Common/interface/OwnVector.h"

namespace reco {
  class CompositeCandidate;
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
  typedef std::vector<CompositeCandidate> CompositeCandidateCollection;
  /// view of a collection containing candidates
  typedef edm::View<CompositeCandidate> CompositeCandidateView;
  /// persistent reference to an object in a collection of Candidate objects
  typedef edm::Ref<CompositeCandidateCollection> CompositeCandidateRef;
  /// vector of references to objects in the same  collection of Candidate objects
  typedef edm::RefVector<CompositeCandidateCollection> CompositeCandidateRefVector;
  /// reference to a collection of Candidate objects
  typedef edm::RefProd<CompositeCandidateCollection> CompositeCandidateRefProd;
  /// vector of references to objects in the same collection of Candidate objects via base type
  typedef edm::RefToBaseProd<CompositeCandidate> CompositeCandidateBaseRefProd;
}  // namespace reco

#endif
