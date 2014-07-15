#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class GsfEleHadronicOverEMCut : public CutApplicatorBase {
public:
  GsfEleHadronicOverEMCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _hadronicOverEMCutValue(c.getParameter<double>("hadronicOverEMCutValue")) {
  }
  
  result_type operator()(const reco::GsfElectronRef&) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  const float _hadronicOverEMCutValue;  
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleHadronicOverEMCut,
		  "GsfEleHadronicOverEMCut");

CutApplicatorBase::result_type 
GsfEleHadronicOverEMCut::
operator()(const reco::GsfElectronRef& cand) const{ 
  return cand->hadronicOverEm() < _hadronicOverEMCutValue;
}
