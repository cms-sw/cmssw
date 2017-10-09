#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class GsfEleSCEtaMultiRangeCut : public CutApplicatorBase {
public:
  GsfEleSCEtaMultiRangeCut(const edm::ParameterSet& c) :
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
  
  result_type operator()(const reco::GsfElectronPtr&) const override final;

  double value(const reco::CandidatePtr& cand) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  const bool _absEta;
  std::vector<std::pair<double,double> > _ranges; 
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleSCEtaMultiRangeCut,
		  "GsfEleSCEtaMultiRangeCut");

CutApplicatorBase::result_type 
GsfEleSCEtaMultiRangeCut::
operator()(const reco::GsfElectronPtr& cand) const{
  const reco::SuperClusterRef& scref = cand->superCluster();
  const double the_eta = ( _absEta ? std::abs(scref->eta()) : scref->eta() );
  bool result = false;
  for(const auto& range : _ranges ) {
    if( the_eta >= range.first && the_eta < range.second ) {
      result = true; break;
    }
  }
  return result;
}

double GsfEleSCEtaMultiRangeCut::value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);
  const reco::SuperClusterRef& scref = ele->superCluster();
  return ( _absEta ? std::abs(scref->eta()) : scref->eta() );
}
