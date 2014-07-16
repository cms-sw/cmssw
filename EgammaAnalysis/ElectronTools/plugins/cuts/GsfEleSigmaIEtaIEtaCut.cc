#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class GsfEleSimgaIEtaIEtaCut : public CutApplicatorBase {
public:
  GsfEleSimgaIEtaIEtaCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _sigmaIEtaIEtaCutValueEB(c.getParameter<double>("sigmaIEtaIEtaCutValue")),
    _sigmaIEtaIEtaCutValueEE(c.getParameter<double>("sigmaIEtaIEtaCutValue")),
    _barrelCutOff(c.getParameter<double>("barrelCutOff")) {
  }
  
  result_type operator()(const reco::GsfElectronRef&) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  const double _sigmaIEtaIEtaCutValueEB,_sigmaIEtaIEtaCutValueEE,_barrelCutOff;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleSimgaIEtaIEtaCut,
		  "GsfEleSimgaIEtaIEtaCut");

CutApplicatorBase::result_type 
GsfEleSimgaIEtaIEtaCut::
operator()(const reco::GsfElectronRef& cand) const{  
  const float sigmaIEtaIEtaCutValue = 
    ( std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ? 
      _sigmaIEtaIEtaCutValueEB : _sigmaIEtaIEtaCutValueEE );
  return cand->sigmaIetaIeta() < sigmaIEtaIEtaCutValue;
}
