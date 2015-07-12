#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

class PhoFull5x5SigmaIEtaIEtaCut : public CutApplicatorBase {
public:
  PhoFull5x5SigmaIEtaIEtaCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::PhotonPtr&) const override final;

  double value(const reco::CandidatePtr& cand) const override final;

  CandidateType candidateType() const override final { 
    return PHOTON; 
  }

private:
  float _cutValueEB;
  float _cutValueEE;
  float _barrelCutOff;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  PhoFull5x5SigmaIEtaIEtaCut,
		  "PhoFull5x5SigmaIEtaIEtaCut");

PhoFull5x5SigmaIEtaIEtaCut::PhoFull5x5SigmaIEtaIEtaCut(const edm::ParameterSet& c) :
  CutApplicatorBase(c),
  _cutValueEB(c.getParameter<double>("cutValueEB")),
  _cutValueEE(c.getParameter<double>("cutValueEE")),
  _barrelCutOff(c.getParameter<double>("barrelCutOff")) {
}

CutApplicatorBase::result_type 
PhoFull5x5SigmaIEtaIEtaCut::
operator()(const reco::PhotonPtr& cand) const{  

  // Figure out the cut value
  const float full5x5SigmaIEtaIEtaCutValue = 
    ( std::abs(cand->superCluster()->eta()) < _barrelCutOff ? 
      _cutValueEB : _cutValueEE );
  
  // Apply the cut and return the result
  return cand->full5x5_sigmaIetaIeta() < full5x5SigmaIEtaIEtaCutValue;
}

double PhoFull5x5SigmaIEtaIEtaCut::
value(const reco::CandidatePtr& cand) const {
  reco::PhotonPtr pho(cand);
  return pho->full5x5_sigmaIetaIeta();
}
