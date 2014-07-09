#ifndef PhysicsTools_SelectorUtils_CandidateCut_h
#define PhysicsTools_SelectorUtils_CandidateCut_h

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace candidate_functions {
  class CandidateCut : public std::unary_function<reco::Candidate,bool>{
  public:
    virtual result_type operator()(const argument_type&) const = 0;
    virtual ~CandidateCut() {}
  };
}

#endif
