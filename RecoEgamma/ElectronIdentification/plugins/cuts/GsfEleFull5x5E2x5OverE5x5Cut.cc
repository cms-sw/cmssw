#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEgamma/EgammaTools/interface/EBEECutValues.h"

class GsfEleFull5x5E2x5OverE5x5Cut : public CutApplicatorBase {
public:
  GsfEleFull5x5E2x5OverE5x5Cut(const edm::ParameterSet& c);

  result_type operator()(const reco::GsfElectronPtr&) const final;

  double value(const reco::CandidatePtr& cand) const final;

  CandidateType candidateType() const final { return ELECTRON; }

private:
  EBEECutValues minE1x5OverE5x5Cut_;
  EBEECutValues minE2x5OverE5x5Cut_;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory, GsfEleFull5x5E2x5OverE5x5Cut, "GsfEleFull5x5E2x5OverE5x5Cut");

GsfEleFull5x5E2x5OverE5x5Cut::GsfEleFull5x5E2x5OverE5x5Cut(const edm::ParameterSet& params)
    : CutApplicatorBase(params),
      minE1x5OverE5x5Cut_(params, "minE1x5OverE5x5"),
      minE2x5OverE5x5Cut_(params, "minE2x5OverE5x5") {}

CutApplicatorBase::result_type GsfEleFull5x5E2x5OverE5x5Cut::operator()(const reco::GsfElectronPtr& cand) const {
  const double e5x5 = cand->full5x5_e5x5();
  const double e2x5OverE5x5 = e5x5 != 0 ? cand->full5x5_e2x5Max() / e5x5 : 0;
  const double e1x5OverE5x5 = e5x5 != 0 ? cand->full5x5_e1x5() / e5x5 : 0;

  return e1x5OverE5x5 > minE1x5OverE5x5Cut_(cand) || e2x5OverE5x5 > minE2x5OverE5x5Cut_(cand);
}

double GsfEleFull5x5E2x5OverE5x5Cut::value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);
  //btw we broke somebodies nice model of assuming every cut is 1D....
  //what this is returning is fairly meaningless...
  return ele->full5x5_e1x5() ? ele->full5x5_e2x5Max() / ele->full5x5_e1x5() : 0.;
}
