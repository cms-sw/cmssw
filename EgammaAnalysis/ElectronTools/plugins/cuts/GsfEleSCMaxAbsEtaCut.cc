#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class GsfEleSCMaxAbsEtaCut : public CutApplicatorBase {
public:
  GsfEleSCMaxAbsEtaCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _maxEta(c.getParameter<bool>("maxEta")) { }
  
  result_type operator()(const reco::GsfElectronRef& cand) const override final {
    const reco::SuperClusterRef& scref = cand->superCluster();
    return std::abs(scref->eta()) < _maxEta;
  }
  
  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  const double _maxEta;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleSCMaxAbsEtaCut,
		  "GsfEleSCMaxAbsEtaCut");

