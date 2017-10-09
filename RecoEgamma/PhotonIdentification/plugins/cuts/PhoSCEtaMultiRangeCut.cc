#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

class PhoSCEtaMultiRangeCut : public CutApplicatorBase {
public:
  PhoSCEtaMultiRangeCut(const edm::ParameterSet& c) :
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
  
  result_type operator()(const reco::PhotonPtr&) const final;

  double value(const reco::CandidatePtr& cand) const final;

  CandidateType candidateType() const final { 
    return PHOTON; 
  }

private:
  const bool _absEta;
  std::vector<std::pair<double,double> > _ranges; 
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  PhoSCEtaMultiRangeCut,
		  "PhoSCEtaMultiRangeCut");

CutApplicatorBase::result_type 
PhoSCEtaMultiRangeCut::
operator()(const reco::PhotonPtr& cand) const{
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

double PhoSCEtaMultiRangeCut::
value(const reco::CandidatePtr& cand) const {
  reco::PhotonPtr pho(cand);
  const reco::SuperClusterRef& scref = pho->superCluster();
  return ( _absEta ? std::abs(scref->eta()) : scref->eta() );
}
