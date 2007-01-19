#ifndef Candidate_CandidateFwd_h
#define Candidate_CandidateFwd_h
#include "DataFormats/Common/interface/OwnVector.h"

namespace reco {
  class Candidate;
}

namespace edm {
  namespace helpers {
    template<typename T> struct PostReadFixupTrait;
    
    template<>
    struct PostReadFixupTrait<reco::Candidate> {
      typedef PostReadFixup type;
    };
  }
}

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  /// collection of Candidate objects
  typedef edm::OwnVector<Candidate> CandidateCollection;
  /// persistent reference to an object in a collection of Candidate objects
  typedef edm::Ref<CandidateCollection> CandidateRef;
  /// persistent reference to an object in a collection of Candidate objects
  typedef edm::RefToBase<Candidate> CandidateBaseRef;
  /// vector of references to objects in the same  collection of Candidate objects
  typedef edm::RefVector<CandidateCollection> CandidateRefVector;
  /// reference to a collection of Candidate objects
  typedef edm::RefProd<CandidateCollection> CandidateRefProd;
  /// iterator over a vector of references Candidate objects
  typedef CandidateRefVector::iterator candidate_iterator;
}

#endif
