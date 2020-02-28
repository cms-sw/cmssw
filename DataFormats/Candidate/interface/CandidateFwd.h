#ifndef Candidate_CandidateFwd_h
#define Candidate_CandidateFwd_h
#include "DataFormats/Common/interface/OwnVector.h"

namespace reco {
  class Candidate;
}

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include "DataFormats/Common/interface/View.h"

namespace reco {
  /// collection of Candidate objects
  typedef edm::OwnVector<Candidate> CandidateCollection;
  /// view of a collection containing candidates
  typedef edm::View<Candidate> CandidateView;
  /// persistent reference to an object in a collection of Candidate objects
  typedef edm::Ptr<Candidate> CandidatePtr;
  /// persistent reference to an object in a collection of Candidate objects
  typedef edm::PtrVector<Candidate> CandidatePtrVector;
  /// persistent reference to an object in a collection of Candidate objects
  typedef edm::Ref<CandidateCollection> CandidateRef;
  /// persistent reference to an object in a collection of Candidate objects
  typedef edm::RefToBase<Candidate> CandidateBaseRef;
  /// vector of references to objects in the same  collection of Candidate objects
  typedef edm::RefVector<CandidateCollection> CandidateRefVector;
  /// vector of references to objects in the same collection of Candidate objects via base type
  typedef edm::RefToBaseVector<Candidate> CandidateBaseRefVector;
  /// reference to a collection of Candidate objects
  typedef edm::RefProd<CandidateCollection> CandidateRefProd;
  /// vector of references to objects in the same collection of Candidate objects via base type
  typedef edm::RefToBaseProd<Candidate> CandidateBaseRefProd;
}  // namespace reco

#endif
