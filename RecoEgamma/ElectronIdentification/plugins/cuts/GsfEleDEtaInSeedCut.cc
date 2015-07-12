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

  double value(const reco::CandidatePtr& cand) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  const double _dEtaInSeedCutValueEB,_dEtaInSeedCutValueEE,_barrelCutOff;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleDEtaInSeedCut,
		  "GsfEleDEtaInSeedCut");

//a little temporary 72X fix
float dEtaInSeed(const reco::GsfElectronPtr& ele){
  return ele->superCluster().isNonnull() && ele->superCluster()->seed().isNonnull() ? 
    ele->deltaEtaSuperClusterTrackAtVtx() - ele->superCluster()->eta() + ele->superCluster()->seed()->eta() : std::numeric_limits<float>::max();
}

CutApplicatorBase::result_type 
GsfEleDEtaInSeedCut::
operator()(const reco::GsfElectronPtr& cand) const{  
  const float dEtaInSeedCutValue = 
    ( std::abs(cand->superCluster()->eta()) < _barrelCutOff ? 
      _dEtaInSeedCutValueEB : _dEtaInSeedCutValueEE );
  // return std::abs(cand->deltaEtaSeedClusterTrackAtVtx()) < dEtaInSeedCutValue;
  return std::abs(dEtaInSeed(cand))<dEtaInSeedCutValue;
}

double GsfEleDEtaInSeedCut::value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);  
  return std::abs(dEtaInSeed(ele));
}
