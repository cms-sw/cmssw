#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

class PatEleSCMaxAbsEtaCut : public CutApplicatorBase {
public:
  PatEleSCMaxAbsEtaCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _maxEta(c.getParameter<bool>("maxEta")) { }
  
  result_type operator()(const pat::Electron& cand) const override final {
    const reco::SuperClusterRef& scref = cand.superCluster();
    return std::abs(scref->eta()) < _maxEta;
  }
  
  CandidateType candidateType() const override final { 
    return PATELECTRON; 
  }

private:
  const double _maxEta;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  PatEleSCMaxAbsEtaCut,
		  "PatEleSCMaxAbsEtaCut");

