#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEgamma/EgammaTools/interface/EffectiveAreas.h"


class GsfEleRelPFIsoScaledCut : public CutApplicatorWithEventContentBase {
  public:
    GsfEleRelPFIsoScaledCut(const edm::ParameterSet& c);
    
    result_type operator()(const reco::GsfElectronPtr&) const final;

    void setConsumes(edm::ConsumesCollector&) final;
    void getEventContent(const edm::EventBase&) final;

    double value(const reco::CandidatePtr& cand) const final;

    CandidateType candidateType() const final { 
      return ELECTRON; 
    }

  private:
    const float barrelC0_, endcapC0_, barrelCpt_, endcapCpt_, barrelCutOff_;
    EffectiveAreas effectiveAreas_;
    edm::Handle<double> rhoHandle_;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory, GsfEleRelPFIsoScaledCut, "GsfEleRelPFIsoScaledCut");

GsfEleRelPFIsoScaledCut::GsfEleRelPFIsoScaledCut(const edm::ParameterSet& c) :
  CutApplicatorWithEventContentBase(c),
  barrelC0_(c.getParameter<double>("barrelC0")),
  endcapC0_(c.getParameter<double>("endcapC0")),
  barrelCpt_(c.getParameter<double>("barrelCpt")),
  endcapCpt_(c.getParameter<double>("endcapCpt")),
  barrelCutOff_(c.getParameter<double>("barrelCutOff")),
  effectiveAreas_((c.getParameter<edm::FileInPath>("effAreasConfigFile")).fullPath())
{
  edm::InputTag rhoTag = c.getParameter<edm::InputTag>("rho");    
  contentTags_.emplace("rho",rhoTag);  
}

void GsfEleRelPFIsoScaledCut::setConsumes(edm::ConsumesCollector& cc){
  auto rho = cc.consumes<double>(contentTags_["rho"]);
  contentTokens_.emplace("rho", rho);
}

void GsfEleRelPFIsoScaledCut::getEventContent(const edm::EventBase& ev){
  ev.getByLabel(contentTags_["rho"], rhoHandle_);
}

CutApplicatorBase::result_type GsfEleRelPFIsoScaledCut::operator()(const reco::GsfElectronPtr& cand) const {  
  // Establish the cut value
  double absEta = std::abs(cand->superCluster()->eta());

  const float C0     = (absEta < barrelCutOff_ ? barrelC0_  : endcapC0_);
  const float Cpt    = (absEta < barrelCutOff_ ? barrelCpt_ : endcapCpt_);
  const float isoCut = C0+Cpt/cand->pt();

  return value(cand) < isoCut;
}

double GsfEleRelPFIsoScaledCut::value(const reco::CandidatePtr& cand) const {

  // Establish the cut value
  reco::GsfElectronPtr ele(cand);
  double absEta = std::abs(ele->superCluster()->eta());
  
  // Compute the combined isolation with effective area correction
  auto pfIso = ele->pfIsolationVariables();
  const float chad = pfIso.sumChargedHadronPt;
  const float nhad = pfIso.sumNeutralHadronEt;
  const float pho  = pfIso.sumPhotonEt;
  const float  eA  = effectiveAreas_.getEffectiveArea(absEta);
  const float rho  = rhoHandle_.isValid() ? (float)(*rhoHandle_) : 0; // std::max likes float arguments
  const float iso  = chad + std::max(0.0f, nhad + pho - rho*eA);
  return iso/cand->pt();
}
