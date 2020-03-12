#ifndef PhysicsTools_SelectorUtils_CandidateCut_h
#define PhysicsTools_SelectorUtils_CandidateCut_h

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace candidate_functions {
  class CandidateCut {
  public:
    using argument_type = reco::CandidatePtr;
    using result_type = bool;

    CandidateCut() {}
    virtual result_type operator()(const argument_type&) const = 0;
    virtual ~CandidateCut() {}

    virtual double value(const reco::CandidatePtr&) const = 0;

    virtual const std::string& name() const = 0;
  };
}  // namespace candidate_functions

#endif
