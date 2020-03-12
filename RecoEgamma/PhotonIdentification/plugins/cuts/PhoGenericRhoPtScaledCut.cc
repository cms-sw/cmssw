#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "RecoEgamma/EgammaTools/interface/EffectiveAreas.h"
#include "RecoEgamma/EgammaTools/interface/ThreadSafeStringCut.h"
#include "RecoEgamma/EgammaTools/interface/EBEECutValues.h"

class PhoGenericRhoPtScaledCut : public CutApplicatorWithEventContentBase {
public:
  PhoGenericRhoPtScaledCut(const edm::ParameterSet& c);

  result_type operator()(const reco::PhotonPtr&) const final;

  void setConsumes(edm::ConsumesCollector&) final;
  void getEventContent(const edm::EventBase&) final;

  double value(const reco::CandidatePtr& cand) const final;

  CandidateType candidateType() const final { return PHOTON; }

private:
  ThreadSafeStringCut<StringObjectFunction<reco::Photon>, reco::Photon> varFunc_;
  bool lessThan_;
  //cut value is constTerm + linearRhoTerm_*rho + linearPtTerm*pt + quadraticPtTerm*pt*pt
  //note EBEECutValues & Effective areas are conceptually the same thing, both are eta
  //binned constants, just Effective areas have arbitary rather than barrel/endcap binnng
  EBEECutValues constTerm_;
  EffectiveAreas linearRhoTerm_;
  EBEECutValues linearPtTerm_;
  EBEECutValues quadraticPtTerm_;

  edm::Handle<double> rhoHandle_;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory, PhoGenericRhoPtScaledCut, "PhoGenericRhoPtScaledCut");

PhoGenericRhoPtScaledCut::PhoGenericRhoPtScaledCut(const edm::ParameterSet& params)
    : CutApplicatorWithEventContentBase(params),
      varFunc_(params.getParameter<std::string>("cutVariable")),
      lessThan_(params.getParameter<bool>("lessThan")),
      constTerm_(params, "constTerm"),
      linearRhoTerm_(params.getParameter<edm::FileInPath>("effAreasConfigFile").fullPath()),
      linearPtTerm_(params, "linearPtTerm"),
      quadraticPtTerm_(params, "quadPtTerm") {
  edm::InputTag rhoTag = params.getParameter<edm::InputTag>("rho");
  contentTags_.emplace("rho", rhoTag);
}

void PhoGenericRhoPtScaledCut::setConsumes(edm::ConsumesCollector& cc) {
  auto rho = cc.consumes<double>(contentTags_["rho"]);
  contentTokens_.emplace("rho", rho);
}

void PhoGenericRhoPtScaledCut::getEventContent(const edm::EventBase& ev) {
  ev.getByLabel(contentTags_["rho"], rhoHandle_);
}

CutApplicatorBase::result_type PhoGenericRhoPtScaledCut::operator()(const reco::PhotonPtr& pho) const {
  const double rho = (*rhoHandle_);

  const float var = varFunc_(*pho);

  const float et = pho->et();
  const float absEta = std::abs(pho->superCluster()->eta());
  const float cutValue = constTerm_(pho) + linearRhoTerm_.getEffectiveArea(absEta) * rho + linearPtTerm_(pho) * et +
                         quadraticPtTerm_(pho) * et * et;
  if (lessThan_)
    return var < cutValue;
  else
    return var >= cutValue;
}

double PhoGenericRhoPtScaledCut::value(const reco::CandidatePtr& cand) const {
  reco::PhotonPtr pho(cand);
  return varFunc_(*pho);
}
