#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class GsfEleHadronicOverEMCut : public CutApplicatorBase {
public:
  GsfEleHadronicOverEMCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _hadronicOverEMCutValueEB(c.getParameter<double>("hadronicOverEMCutValueEB")),
    _hadronicOverEMCutValueEE(c.getParameter<double>("hadronicOverEMCutValueEE")),
    _barrelCutOff(c.getParameter<double>("barrelCutOff")) {
  }
  
  result_type operator()(const reco::GsfElectronRef&) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  const float _hadronicOverEMCutValueEB, _hadronicOverEMCutValueEE, _barrelCutOff;  
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleHadronicOverEMCut,
		  "GsfEleHadronicOverEMCut");

CutApplicatorBase::result_type 
GsfEleHadronicOverEMCut::
operator()(const reco::GsfElectronRef& cand) const { 
  const float hadronicOverEMCutValue = 
    ( std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ? 
      _hadronicOverEMCutValueEB : _hadronicOverEMCutValueEE );
  return cand->hadronicOverEm() < hadronicOverEMCutValue;
}
