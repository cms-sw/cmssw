#ifndef Candidate_CandidateFwd_h
#define Candidate_CandidateFwd_h
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class Candidate;
  typedef edm::OwnVector<Candidate> CandidateCollection;
  typedef edm::Ref<CandidateCollection> CandidateRef;
  typedef edm::RefVector<CandidateCollection> CandidateRefs;
  typedef edm::RefProd<CandidateCollection> CandidatesRef;
  typedef CandidateRefs::iterator candidate_iterator;
}

#endif
