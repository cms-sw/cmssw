#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class GsfEleDPhiInCut : public CutApplicatorBase {
public:
  GsfEleDPhiInCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _dPhiInCutValue(c.getParameter<double>("dPhiInCutValue")) {    
  }
  
  result_type operator()(const reco::GsfElectron&) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  const double _dPhiInCutValue;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleDPhiInCut,
		  "GsfEleDPhiInCut");

CutApplicatorBase::result_type 
GsfEleDPhiInCut::
operator()(const reco::GsfElectron& cand) const{  
  return cand.deltaPhiSuperClusterTrackAtVtx() < _dPhiInCutValue;
}
