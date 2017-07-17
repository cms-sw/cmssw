#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"

class EtaMultiRangeCut : public CutApplicatorBase {
public:
  EtaMultiRangeCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _absEta(c.getParameter<bool>("useAbsEta")) {
    const std::vector<edm::ParameterSet>& ranges =
      c.getParameterSetVector("allowedEtaRanges");
    for( const auto& range : ranges ) {
      const double min = range.getParameter<double>("minEta");
      const double max = range.getParameter<double>("maxEta");
      _ranges.emplace_back(min,max);
    }
  }
  
  double value(const reco::CandidatePtr& cand) const override final { 
    return ( _absEta ? std::abs(cand->eta()) : cand->eta() );
  }

  result_type asCandidate(const argument_type&) const override final;

private:
  const bool _absEta;
  std::vector<std::pair<double,double> > _ranges; 
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,EtaMultiRangeCut,"EtaMultiRangeCut");

CutApplicatorBase::result_type 
EtaMultiRangeCut::
asCandidate(const argument_type& cand) const {
  const double the_eta = ( _absEta ? std::abs(cand->eta()) : cand->eta() );
  bool result = false;
  for(const auto& range : _ranges ) {
    if( the_eta >= range.first && the_eta < range.second ) {
      result = true; break;
    }
  }
  return result;
}
