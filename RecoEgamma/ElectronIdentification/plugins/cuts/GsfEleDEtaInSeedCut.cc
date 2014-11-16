#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class GsfEleDEtaInSeedCut : public CutApplicatorBase {
public:
  GsfEleDEtaInSeedCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _dEtaInSeedCutValueEB(c.getParameter<double>("dEtaInSeedCutValueEB")),
    _dEtaInSeedCutValueEE(c.getParameter<double>("dEtaInSeedCutValueEE")),
    _barrelCutOff(c.getParameter<double>("barrelCutOff")){    
  }
  
  result_type operator()(const reco::GsfElectronPtr&) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  const double _dEtaInSeedCutValueEB,_dEtaInSeedCutValueEE,_barrelCutOff;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleDEtaInSeedCut,
		  "GsfEleDEtaInSeedCut");

CutApplicatorBase::result_type 
GsfEleDEtaInSeedCut::
operator()(const reco::GsfElectronPtr& cand) const{  
  const float dEtaInSeedCutValue = 
    ( std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ? 
      _dEtaInSeedCutValueEB : _dEtaInSeedCutValueEE );
  return std::abs(cand->deltaEtaSeedClusterTrackAtVtx()) < dEtaInSeedCutValue;
}
