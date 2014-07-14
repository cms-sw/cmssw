#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class GsfEleDEtaInCut : public CutApplicatorBase {
public:
  GsfEleDEtaInCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _dEtaInCutValue(c.getParameter<double>("dEtaInCutValue")) {    
  }
  
  result_type operator()(const reco::GsfElectron&) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  const double _dEtaInCutValue;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleDEtaInCut,
		  "GsfEleDEtaInCut");

CutApplicatorBase::result_type 
GsfEleDEtaInCut::
operator()(const reco::GsfElectron& cand) const{  
  return cand.deltaEtaSuperClusterTrackAtVtx() < _dEtaInCutValue;
}
