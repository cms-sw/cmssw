#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h" 

class GsfEleMinEcalEtCut : public CutApplicatorBase {
public:
  GsfEleMinEcalEtCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronPtr&) const final;

  double value(const reco::CandidatePtr& cand) const final;

  CandidateType candidateType() const final { 
    return ELECTRON; 
  }

private:  
  const double minEt_;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleMinEcalEtCut,
		  "GsfEleMinEcalEtCut");

GsfEleMinEcalEtCut::GsfEleMinEcalEtCut(const edm::ParameterSet& c) :
  CutApplicatorBase(c),
  minEt_(c.getParameter<double>("minEt"))
{

}

CutApplicatorBase::result_type 
GsfEleMinEcalEtCut::
operator()(const reco::GsfElectronPtr& cand) const{  
  const reco::GsfElectron& ele = *cand;
  const float sinTheta = ele.p()!=0 ? ele.pt()/ele.p() : 0.;
  const float et = ele.ecalEnergy()*sinTheta;
  return et>minEt_;
}

double GsfEleMinEcalEtCut::value(const reco::CandidatePtr& cand) const {
  const reco::GsfElectronPtr elePtr(cand);
  const reco::GsfElectron& ele = *elePtr;
  const float sinTheta = ele.p()!=0 ? ele.pt()/ele.p() : 0.;
  const float et = ele.ecalEnergy()*sinTheta;
  return et;
}
