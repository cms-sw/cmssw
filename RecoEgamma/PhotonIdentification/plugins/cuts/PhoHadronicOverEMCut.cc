#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

class PhoHadronicOverEMCut : public CutApplicatorBase {
public:
  PhoHadronicOverEMCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _hadronicOverEMCutValueEB(c.getParameter<double>("hadronicOverEMCutValueEB")),
    _hadronicOverEMCutValueEE(c.getParameter<double>("hadronicOverEMCutValueEE")),
    _barrelCutOff(c.getParameter<double>("barrelCutOff")) {
  }
  
  result_type operator()(const reco::PhotonPtr&) const override final;

  double value(const reco::CandidatePtr& cand) const override final;

  CandidateType candidateType() const override final { 
    return PHOTON; 
  }

private:
  const float _hadronicOverEMCutValueEB, _hadronicOverEMCutValueEE, _barrelCutOff;  
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  PhoHadronicOverEMCut,
		  "PhoHadronicOverEMCut");

CutApplicatorBase::result_type 
PhoHadronicOverEMCut::
operator()(const reco::PhotonPtr& cand) const { 
  const float hadronicOverEMCutValue = 
    ( std::abs(cand->superCluster()->eta()) < _barrelCutOff ? 
      _hadronicOverEMCutValueEB : _hadronicOverEMCutValueEE );

  return cand->hadronicOverEm() < hadronicOverEMCutValue;
}

double PhoHadronicOverEMCut::
value(const reco::CandidatePtr& cand) const {
  reco::PhotonPtr pho(cand);
  return pho->hadronicOverEm();
}
