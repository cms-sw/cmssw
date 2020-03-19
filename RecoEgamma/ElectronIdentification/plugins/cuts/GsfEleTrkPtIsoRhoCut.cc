#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"
#include "RecoEgamma/EgammaTools/interface/EBEECutValues.h"

class GsfEleTrkPtIsoRhoCut : public CutApplicatorWithEventContentBase {
public:
  GsfEleTrkPtIsoRhoCut(const edm::ParameterSet& c);

  result_type operator()(const reco::GsfElectronPtr&) const final;

  void setConsumes(edm::ConsumesCollector&) final;
  void getEventContent(const edm::EventBase&) final;

  double value(const reco::CandidatePtr& cand) const final;

  CandidateType candidateType() const final { return ELECTRON; }

private:
  EBEECutValues slopeTerm_;
  EBEECutValues slopeStart_;
  EBEECutValues constTerm_;
  EBEECutValues rhoEtStart_;
  EBEECutValues rhoEA_;

  edm::Handle<double> rhoHandle_;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory, GsfEleTrkPtIsoRhoCut, "GsfEleTrkPtIsoRhoCut");

GsfEleTrkPtIsoRhoCut::GsfEleTrkPtIsoRhoCut(const edm::ParameterSet& params)
    : CutApplicatorWithEventContentBase(params),
      slopeTerm_(params, "slopeTerm"),
      slopeStart_(params, "slopeStart"),
      constTerm_(params, "constTerm"),
      rhoEtStart_(params, "rhoEtStart"),
      rhoEA_(params, "rhoEA") {
  edm::InputTag rhoTag = params.getParameter<edm::InputTag>("rho");
  contentTags_.emplace("rho", rhoTag);
}

void GsfEleTrkPtIsoRhoCut::setConsumes(edm::ConsumesCollector& cc) {
  auto rho = cc.consumes<double>(contentTags_["rho"]);
  contentTokens_.emplace("rho", rho);
}

void GsfEleTrkPtIsoRhoCut::getEventContent(const edm::EventBase& ev) { ev.getByLabel(contentTags_["rho"], rhoHandle_); }

CutApplicatorBase::result_type GsfEleTrkPtIsoRhoCut::operator()(const reco::GsfElectronPtr& cand) const {
  const double rho = (*rhoHandle_);
  const float isolTrkPt = cand->dr03TkSumPt();

  const float et = cand->et();
  const float cutValue =
      et > slopeStart_(cand) ? slopeTerm_(cand) * (et - slopeStart_(cand)) + constTerm_(cand) : constTerm_(cand);

  const float rhoCutValue = et >= rhoEtStart_(cand) ? rhoEA_(cand) * rho : 0.;

  return isolTrkPt < cutValue + rhoCutValue;
}

double GsfEleTrkPtIsoRhoCut::value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);
  return ele->dr03TkSumPt();
}
