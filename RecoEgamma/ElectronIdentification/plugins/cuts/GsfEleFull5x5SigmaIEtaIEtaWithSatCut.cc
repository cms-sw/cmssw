#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "RecoEgamma/EgammaTools/interface/EBEECutValues.h"

class GsfEleFull5x5SigmaIEtaIEtaWithSatCut : public CutApplicatorBase {
public:
  GsfEleFull5x5SigmaIEtaIEtaWithSatCut(const edm::ParameterSet& c);

  result_type operator()(const reco::GsfElectronPtr&) const final;

  double value(const reco::CandidatePtr& cand) const final;

  CandidateType candidateType() const final { return ELECTRON; }

private:
  EBEECutValues maxSigmaIEtaIEtaCut_;
  EBEECutValuesInt maxNrSatCrysIn5x5Cut_;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory, GsfEleFull5x5SigmaIEtaIEtaWithSatCut, "GsfEleFull5x5SigmaIEtaIEtaWithSatCut");

GsfEleFull5x5SigmaIEtaIEtaWithSatCut::GsfEleFull5x5SigmaIEtaIEtaWithSatCut(const edm::ParameterSet& params)
    : CutApplicatorBase(params),
      maxSigmaIEtaIEtaCut_(params, "maxSigmaIEtaIEta"),
      maxNrSatCrysIn5x5Cut_(params, "maxNrSatCrysIn5x5") {}

CutApplicatorBase::result_type GsfEleFull5x5SigmaIEtaIEtaWithSatCut::operator()(const reco::GsfElectronPtr& cand) const {
  if (cand->nSaturatedXtals() > maxNrSatCrysIn5x5Cut_(cand))
    return true;
  else
    return cand->full5x5_sigmaIetaIeta() < maxSigmaIEtaIEtaCut_(cand);
}

double GsfEleFull5x5SigmaIEtaIEtaWithSatCut::value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);
  return ele->full5x5_sigmaIetaIeta();
}
