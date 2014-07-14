#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class GsfEleSimgaIEtaIEtaCut : public CutApplicatorBase {
public:
  GsfEleSimgaIEtaIEtaCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _sigmaIEtaIEtaCutValue(c.getParameter<double>("sigmaIEtaIEtaCutValue")) {
  }
  
  result_type operator()(const reco::GsfElectron&) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  const double _sigmaIEtaIEtaCutValue;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleSimgaIEtaIEtaCut,
		  "GsfEleSimgaIEtaIEtaCut");

CutApplicatorBase::result_type 
GsfEleSimgaIEtaIEtaCut::
operator()(const reco::GsfElectron& cand) const{  
  return cand.sigmaIetaIeta() < _sigmaIEtaIEtaCutValue;
}
