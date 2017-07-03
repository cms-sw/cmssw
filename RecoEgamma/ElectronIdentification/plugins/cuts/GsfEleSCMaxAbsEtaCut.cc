#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class GsfEleSCMaxAbsEtaCut : public CutApplicatorBase {
public:
  GsfEleSCMaxAbsEtaCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _maxEta(c.getParameter<bool>("maxEta")) { }
  
  result_type operator()(const reco::GsfElectronPtr& cand) const final {
    const reco::SuperClusterRef& scref = cand->superCluster();
    return std::abs(scref->eta()) < _maxEta;
  }
  
  double value(const reco::CandidatePtr& cand) const final {
    reco::GsfElectronPtr ele(cand);
    const reco::SuperClusterRef& scref = ele->superCluster();
    return std::abs(scref->eta());
  }

  CandidateType candidateType() const final { 
    return ELECTRON; 
  }

private:
  const double _maxEta;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleSCMaxAbsEtaCut,
		  "GsfEleSCMaxAbsEtaCut");

