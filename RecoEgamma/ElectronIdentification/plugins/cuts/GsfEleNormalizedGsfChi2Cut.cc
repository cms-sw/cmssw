#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class GsfEleNormalizedGsfChi2Cut : public CutApplicatorBase {
public:
  GsfEleNormalizedGsfChi2Cut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _normalizedGsfChi2CutValueEB(c.getParameter<double>("normalizedGsfChi2CutValueEB")),
    _normalizedGsfChi2CutValueEE(c.getParameter<double>("normalizedGsfChi2CutValueEE")),
    _barrelCutOff(c.getParameter<double>("barrelCutOff")){    
  }
  
  result_type operator()(const reco::GsfElectronPtr&) const override final;

  double value(const reco::CandidatePtr& cand) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  const double _normalizedGsfChi2CutValueEB,_normalizedGsfChi2CutValueEE,_barrelCutOff;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleNormalizedGsfChi2Cut,
		  "GsfEleNormalizedGsfChi2Cut");

float normalizedGsfChi2(const reco::GsfElectronPtr& ele){
  return ele->gsfTrack().isNonnull() ? ele->gsfTrack()->normalizedChi2() : std::numeric_limits<float>::max();
}

CutApplicatorBase::result_type 
GsfEleNormalizedGsfChi2Cut::
operator()(const reco::GsfElectronPtr& cand) const{  
  const float normalizedGsfChi2CutValue = 
    ( std::abs(cand->superCluster()->eta()) < _barrelCutOff ? 
      _normalizedGsfChi2CutValueEB : _normalizedGsfChi2CutValueEE );

  return std::abs(normalizedGsfChi2(cand)) < normalizedGsfChi2CutValue;
}

double GsfEleNormalizedGsfChi2Cut::value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);  
  return std::abs(normalizedGsfChi2(ele));
}
