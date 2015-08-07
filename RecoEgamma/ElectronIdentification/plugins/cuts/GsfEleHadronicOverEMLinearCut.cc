#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEgamma/ElectronIdentification/interface/EBEECutValues.h"
class GsfEleHadronicOverEMLinearCut : public CutApplicatorBase {
public:
  GsfEleHadronicOverEMLinearCut(const edm::ParameterSet& params) : 
    CutApplicatorBase(params),
    slopeTerm_(params,"slopeTerm"),
    slopeStart_(params,"slopeStart"),
    constTerm_(params,"constTerm"){}
  
  result_type operator()(const reco::GsfElectronPtr&) const override final;

  double value(const reco::CandidatePtr& cand) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  EBEECutValues slopeTerm_;
  EBEECutValues slopeStart_;
  EBEECutValues constTerm_;
   
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleHadronicOverEMLinearCut,
		  "GsfEleHadronicOverEMLinearCut");

CutApplicatorBase::result_type 
GsfEleHadronicOverEMLinearCut::
operator()(const reco::GsfElectronPtr& cand) const { 

  const float energy = cand->superCluster()->energy();
  const float cutValue = energy > slopeStart_(cand)  ? slopeTerm_(cand)*(energy-slopeStart_(cand)) + constTerm_(cand) : constTerm_(cand);

  return cand->hadronicOverEm()*energy < cutValue;
}

double GsfEleHadronicOverEMLinearCut::
value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);
  const float energy = ele->superCluster()->energy();
  return ele->hadronicOverEm()*energy;
}
