#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class GsfEleNormalizedGsfChi2Cut : public CutApplicatorBase {
public:
  GsfEleNormalizedGsfChi2Cut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    normalizedGsfChi2CutValueEB_(c.getParameter<double>("normalizedGsfChi2CutValueEB")),
    normalizedGsfChi2CutValueEE_(c.getParameter<double>("normalizedGsfChi2CutValueEE")),
    barrelCutOff_(c.getParameter<double>("barrelCutOff")){    
  }
  
  result_type operator()(const reco::GsfElectronPtr&) const final;

  double value(const reco::CandidatePtr& cand) const final;

  CandidateType candidateType() const final { 
    return ELECTRON; 
  }

private:
  const double normalizedGsfChi2CutValueEB_,normalizedGsfChi2CutValueEE_,barrelCutOff_;
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
    ( std::abs(cand->superCluster()->eta()) < barrelCutOff_ ? 
      normalizedGsfChi2CutValueEB_ : normalizedGsfChi2CutValueEE_ );

  return std::abs(normalizedGsfChi2(cand)) < normalizedGsfChi2CutValue;
}

double GsfEleNormalizedGsfChi2Cut::value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);  
  return std::abs(normalizedGsfChi2(ele));
}
