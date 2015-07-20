#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class GsfEleDPhiInCut : public CutApplicatorBase {
public:
  GsfEleDPhiInCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _dPhiInCutValueEB(c.getParameter<double>("dPhiInCutValueEB")),
    _dPhiInCutValueEE(c.getParameter<double>("dPhiInCutValueEE")),
    _barrelCutOff(c.getParameter<double>("barrelCutOff")) {    
  }
  
  result_type operator()(const reco::GsfElectronPtr&) const override final;

  double value(const reco::CandidatePtr& cand) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  const double _dPhiInCutValueEB, _dPhiInCutValueEE, _barrelCutOff;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleDPhiInCut,
		  "GsfEleDPhiInCut");

CutApplicatorBase::result_type 
GsfEleDPhiInCut::
operator()(const reco::GsfElectronPtr& cand) const{  
  const float dPhiInCutValue = 
    ( std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ? 
      _dPhiInCutValueEB : _dPhiInCutValueEE );
  return std::abs(cand->deltaPhiSuperClusterTrackAtVtx()) < dPhiInCutValue;
}

double GsfEleDPhiInCut::value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);
  return std::abs(ele->deltaPhiSuperClusterTrackAtVtx());
}
