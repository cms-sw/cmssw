#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "CommonTools/Egamma/interface/EffectiveAreas.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "CommonTools/Utils/interface/ThreadSafeFunctor.h"
#include "RecoEgamma/EgammaTools/interface/EBEECutValues.h"

class PhoGenericQuadraticRhoPtScaledCut : public CutApplicatorWithEventContentBase {
public:
  PhoGenericQuadraticRhoPtScaledCut(const edm::ParameterSet& c);

  result_type operator()(const reco::PhotonPtr&) const final;

  void setConsumes(edm::ConsumesCollector&) final;
  void getEventContent(const edm::EventBase&) final;

  double value(const reco::CandidatePtr& cand) const final;

  CandidateType candidateType() const final { return PHOTON; }

private:
  ThreadSafeFunctor<StringObjectFunction<reco::Photon>> varFunc_;
  bool lessThan_;
  //cut value is constTerm + linearRhoTerm_*rho + linearPtTerm*pt + quadraticPtTerm*pt*pt
  //note EBEECutValues & Effective areas are conceptually the same thing, both are eta
  //binned constants, just Effective areas have arbitary rather than barrel/endcap binnng
  EBEECutValues constTerm_;
  EffectiveAreas rhoEA_;
  EBEECutValues linearPtTerm_;
  EBEECutValues quadraticPtTerm_;

  edm::Handle<double> rhoHandle_;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory, PhoGenericQuadraticRhoPtScaledCut, "PhoGenericQuadraticRhoPtScaledCut");

PhoGenericQuadraticRhoPtScaledCut::PhoGenericQuadraticRhoPtScaledCut(const edm::ParameterSet& params)
    : CutApplicatorWithEventContentBase(params),
      varFunc_(params.getParameter<std::string>("cutVariable")),
      lessThan_(params.getParameter<bool>("lessThan")),
      constTerm_(params, "constTerm"),
      rhoEA_(params.getParameter<edm::FileInPath>("effAreasConfigFile").fullPath(),
             params.getParameter<bool>("quadEAflag")),
      linearPtTerm_(params, "linearPtTerm"),
      quadraticPtTerm_(params, "quadraticPtTerm") {
  edm::InputTag rhoTag = params.getParameter<edm::InputTag>("rho");
  contentTags_.emplace("rho", rhoTag);
}

void PhoGenericQuadraticRhoPtScaledCut::setConsumes(edm::ConsumesCollector& cc) {
  auto rho = cc.consumes<double>(contentTags_["rho"]);
  contentTokens_.emplace("rho", rho);
}

void PhoGenericQuadraticRhoPtScaledCut::getEventContent(const edm::EventBase& ev) {
  ev.getByLabel(contentTags_["rho"], rhoHandle_);
}

CutApplicatorBase::result_type PhoGenericQuadraticRhoPtScaledCut::operator()(const reco::PhotonPtr& pho) const {
  const double rho = (*rhoHandle_);

  const float var = varFunc_(*pho);

  const float et = pho->et();
  const float absEta = std::abs(pho->superCluster()->eta());
  const float cutValue = constTerm_(pho) + rhoEA_.getLinearEA(absEta) * rho +
                         rhoEA_.getQuadraticEA(absEta) * rho * rho + linearPtTerm_(pho) * et +
                         quadraticPtTerm_(pho) * et * et;
  if (lessThan_)
    return var < cutValue;
  else
    return var >= cutValue;
}

double PhoGenericQuadraticRhoPtScaledCut::value(const reco::CandidatePtr& cand) const {
  reco::PhotonPtr pho(cand);
  return varFunc_(*pho);
}
