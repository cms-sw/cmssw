#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class GsfEleDEtaInCut : public CutApplicatorBase {
public:
  GsfEleDEtaInCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _dEtaInCutValueEB(c.getParameter<double>("dEtaInCutValueEB")),
    _dEtaInCutValueEE(c.getParameter<double>("dEtaInCutValueEE")),
    _barrelCutOff(c.getParameter<double>("barrelCutOff")){    
  }
  
  result_type operator()(const reco::GsfElectronPtr&) const final;

  double value(const reco::CandidatePtr& cand) const final;

  CandidateType candidateType() const final { 
    return ELECTRON; 
  }

private:
  const double _dEtaInCutValueEB,_dEtaInCutValueEE,_barrelCutOff;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleDEtaInCut,
		  "GsfEleDEtaInCut");

CutApplicatorBase::result_type 
GsfEleDEtaInCut::
operator()(const reco::GsfElectronPtr& cand) const{  
  const float dEtaInCutValue = 
    ( std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ? 
      _dEtaInCutValueEB : _dEtaInCutValueEE );
  return std::abs(cand->deltaEtaSuperClusterTrackAtVtx()) < dEtaInCutValue;
}

double GsfEleDEtaInCut::value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);
  return std::abs(ele->deltaEtaSuperClusterTrackAtVtx());
}
