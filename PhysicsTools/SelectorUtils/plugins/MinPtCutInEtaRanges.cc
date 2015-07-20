#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"

class MinPtCutInEtaRanges : public CutApplicatorBase {
public:
  MinPtCutInEtaRanges(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _absEta(c.getParameter<bool>("useAbsEta")) {
    const std::vector<edm::ParameterSet>& ranges =
      c.getParameterSetVector("allowedEtaRanges");
    for( const auto& range : ranges ) {
      const double minEta = range.getParameter<double>("minEta");
      const double maxEta = range.getParameter<double>("maxEta");
      const double minPt = range.getParameter<double>("minPt");
      _ranges.emplace_back(minEta,maxEta);
      _minPt.push_back(minPt);
    }
  }

  double value(const reco::CandidatePtr& cand) const override final {
    return cand->pt();
  }

  result_type asCandidate(const argument_type&) const override final;

private:
  const bool _absEta;
  std::vector<std::pair<double,double> > _ranges; 
  std::vector<double> _minPt; // indexed as above
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,MinPtCutInEtaRanges,
		  "MinPtCutInEtaRanges");

CutApplicatorBase::result_type 
MinPtCutInEtaRanges::
asCandidate(const argument_type& cand) const{
  const double the_eta = ( _absEta ? std::abs(cand->eta()) : cand->eta() );
  bool result = false;
  for( unsigned i = 0; i < _ranges.size(); ++i ) {
    const auto& range = _ranges[i];
    if( the_eta >= range.first && the_eta < range.second && 
	cand->pt() > _minPt[i] ) {
      result = true; break;
    }
  }
  return result;
}
