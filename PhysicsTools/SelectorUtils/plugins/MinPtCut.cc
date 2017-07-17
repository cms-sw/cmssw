#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"

class MinPtCut : public CutApplicatorBase {
public:
  MinPtCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _minPt(c.getParameter<double>("minPt")) { }
  
  double value(const reco::CandidatePtr& cand) const override final {
    return cand->pt();
  }

  result_type asCandidate(const argument_type& cand) const override final {
    return cand->pt() > _minPt;
  }

private:
  const double _minPt;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,MinPtCut,"MinPtCut");

