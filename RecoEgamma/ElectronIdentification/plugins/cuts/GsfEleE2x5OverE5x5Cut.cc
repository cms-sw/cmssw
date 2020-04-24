#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEgamma/ElectronIdentification/interface/EBEECutValues.h"

class GsfEleE2x5OverE5x5Cut : public CutApplicatorBase {
public:
  GsfEleE2x5OverE5x5Cut(const edm::ParameterSet& params) :
    CutApplicatorBase(params),
    minE1x5OverE5x5_(params,"minE1x5OverE5x5"),
    minE2x5OverE5x5_(params,"minE2x5OverE5x5"){}
  
  result_type operator()(const reco::GsfElectronPtr&) const final;

  double value(const reco::CandidatePtr& cand) const final;
  
  CandidateType candidateType() const final { 
    return ELECTRON; 
  }

private:
  EBEECutValues minE1x5OverE5x5_;
  EBEECutValues minE2x5OverE5x5_;
  

};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleE2x5OverE5x5Cut,
		  "GsfEleE2x5OverE5x5Cut");

CutApplicatorBase::result_type 
GsfEleE2x5OverE5x5Cut::
operator()(const reco::GsfElectronPtr& cand) const{
  
  return cand->e2x5Max() > minE2x5OverE5x5_(cand)*cand->e5x5() || 
         cand->e1x5()    > minE1x5OverE5x5_(cand)*cand->e5x5();
  
}

double GsfEleE2x5OverE5x5Cut::value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);
  //btw we broke somebodies nice model of assuming every cut is 1D....
  //what this is returning is fairly meaningless...
  return ele->e1x5() ? ele->e2x5Max()/ele->e1x5() : 0.;
}
