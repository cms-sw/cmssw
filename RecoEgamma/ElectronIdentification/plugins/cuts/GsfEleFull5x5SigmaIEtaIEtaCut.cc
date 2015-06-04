#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class GsfEleFull5x5SigmaIEtaIEtaCut : public CutApplicatorBase {
public:
  GsfEleFull5x5SigmaIEtaIEtaCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronPtr&) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  float _full5x5SigmaIEtaIEtaCutValueEB;
  float _full5x5SigmaIEtaIEtaCutValueEE;
  float _barrelCutOff;

};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleFull5x5SigmaIEtaIEtaCut,
		  "GsfEleFull5x5SigmaIEtaIEtaCut");

GsfEleFull5x5SigmaIEtaIEtaCut::GsfEleFull5x5SigmaIEtaIEtaCut(const edm::ParameterSet& c) :
  CutApplicatorBase(c),
  _full5x5SigmaIEtaIEtaCutValueEB(c.getParameter<double>("full5x5SigmaIEtaIEtaCutValueEB")),
  _full5x5SigmaIEtaIEtaCutValueEE(c.getParameter<double>("full5x5SigmaIEtaIEtaCutValueEE")),
  _barrelCutOff(c.getParameter<double>("barrelCutOff")) {
  
}


CutApplicatorBase::result_type 
GsfEleFull5x5SigmaIEtaIEtaCut::
operator()(const reco::GsfElectronPtr& cand) const{  

  // Figure out the cut value
  const float full5x5SigmaIEtaIEtaCutValue = 
    ( std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ? 
      _full5x5SigmaIEtaIEtaCutValueEB : _full5x5SigmaIEtaIEtaCutValueEE );
  
  // Retrieve the variable value for this particle
  const float full5x5SigmaIEtaIEta = cand->full5x5_sigmaIetaIeta();
  
  // Apply the cut and return the result
  return full5x5SigmaIEtaIEta < full5x5SigmaIEtaIEtaCutValue;
}
