#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEgamma/ElectronIdentification/interface/EBEECutValues.h"

class GsfEleE2x5OverE5x5Cut : public CutApplicatorBase {
public:
  GsfEleE2x5OverE5x5Cut(const edm::ParameterSet& params) :
    CutApplicatorBase(params),
    minE1x5OverE5x5_(params,"minE1x5OverE5x5"),
    minE2x5OverE5x5_(params,"minE2x5OverE5x5"){}
  
  result_type operator()(const reco::GsfElectronPtr&) const override final;

  double value(const reco::CandidatePtr& cand) const override final;
  
  CandidateType candidateType() const override final { 
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
  // here return the ratio since it is a decent encoding of the value of 
  // both variables
  return ele->e2x5Max()/ele->e1x5();
}
