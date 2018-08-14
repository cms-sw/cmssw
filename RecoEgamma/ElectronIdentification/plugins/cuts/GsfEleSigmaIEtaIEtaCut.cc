#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class GsfEleSigmaIEtaIEtaCut : public CutApplicatorBase {
public:
  GsfEleSigmaIEtaIEtaCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _sigmaIEtaIEtaCutValueEB(c.getParameter<double>("sigmaIEtaIEtaCutValueEB")),
    _sigmaIEtaIEtaCutValueEE(c.getParameter<double>("sigmaIEtaIEtaCutValueEE")),
    _barrelCutOff(c.getParameter<double>("barrelCutOff")) {
  }
  
  result_type operator()(const reco::GsfElectronPtr&) const final;

  double value(const reco::CandidatePtr& cand) const final;

  CandidateType candidateType() const final { 
    return ELECTRON; 
  }

private:
  const double _sigmaIEtaIEtaCutValueEB,_sigmaIEtaIEtaCutValueEE,_barrelCutOff;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleSigmaIEtaIEtaCut,
		  "GsfEleSigmaIEtaIEtaCut");

CutApplicatorBase::result_type 
GsfEleSigmaIEtaIEtaCut::
operator()(const reco::GsfElectronPtr& cand) const{  
  const float sigmaIEtaIEtaCutValue = 
    ( std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ? 
      _sigmaIEtaIEtaCutValueEB : _sigmaIEtaIEtaCutValueEE );
  return cand->sigmaIetaIeta() < sigmaIEtaIEtaCutValue;
}

double GsfEleSigmaIEtaIEtaCut::
value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);  
  return ele->sigmaIetaIeta();
}
