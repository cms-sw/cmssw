#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"

class MaxAbsEtaCut : public CutApplicatorBase {
public:
  MaxAbsEtaCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _maxEta(c.getParameter<double>("maxEta")) { }
  
  double value(const reco::CandidatePtr& cand) const override final {
    return std::abs(cand->eta());
  }

  result_type asCandidate(const argument_type& cand) const override final {
    return std::abs(cand->eta()) < _maxEta;
  }

private:
  const double _maxEta;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,MaxAbsEtaCut,"MaxAbsEtaCut");

