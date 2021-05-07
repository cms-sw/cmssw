#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

template <bool isBC>
class PhotonHcalOverEcalCut : public CutApplicatorBase {
public:
  PhotonHcalOverEcalCut(const edm::ParameterSet& c)
      : CutApplicatorBase(c),
        _hcalOverEcalCutValueEB(c.getParameter<double>("hcalOverEcalCutValueEB")),
        _hcalOverEcalCutValueEE(c.getParameter<double>("hcalOverEcalCutValueEE")),
        _barrelCutOff(c.getParameter<double>("barrelCutOff")) {}

  result_type operator()(const reco::PhotonPtr&) const final;

  double value(const reco::CandidatePtr& cand) const final;

  CandidateType candidateType() const final { return PHOTON; }

private:
  const float _hcalOverEcalCutValueEB, _hcalOverEcalCutValueEE, _barrelCutOff;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory, PhotonHcalOverEcalCut<false>, "PhotonHcalOverEcalCut");
DEFINE_EDM_PLUGIN(CutApplicatorFactory, PhotonHcalOverEcalCut<true>, "PhotonHcalOverEcalBcCut");

template <bool isBC>
CutApplicatorBase::result_type PhotonHcalOverEcalCut<isBC>::operator()(const reco::PhotonPtr& cand) const {
  const float hcalOverEcalCutValue =
      (std::abs(cand->superCluster()->eta()) < _barrelCutOff ? _hcalOverEcalCutValueEB : _hcalOverEcalCutValueEE);

  if constexpr (isBC)
    return cand->hcalOverEcalBc() < hcalOverEcalCutValue;
  else
    return cand->hcalOverEcal() < hcalOverEcalCutValue;
}

template <bool isBC>
double PhotonHcalOverEcalCut<isBC>::value(const reco::CandidatePtr& cand) const {
  reco::PhotonPtr pho(cand);
  if constexpr (isBC)
    return pho->hcalOverEcalBc();
  else
    return pho->hcalOverEcal();
}
