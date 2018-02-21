#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class GsfEleHadronicOverEMEnergyScaledCut : public CutApplicatorWithEventContentBase {
public:
  GsfEleHadronicOverEMEnergyScaledCut(const edm::ParameterSet& c) :
    CutApplicatorWithEventContentBase(c),
    barrelC0_(c.getParameter<double>("barrelC0")),
    barrelCE_(c.getParameter<double>("barrelCE")),
    barrelCr_(c.getParameter<double>("barrelCr")),
    endcapC0_(c.getParameter<double>("endcapC0")),
    endcapCE_(c.getParameter<double>("endcapCE")),
    endcapCr_(c.getParameter<double>("endcapCr")),
    barrelCutOff_(c.getParameter<double>("barrelCutOff")) 
  {
    edm::InputTag rhoTag = c.getParameter<edm::InputTag>("rho");    
    contentTags_.emplace("rho",rhoTag);  
    
  }
  
  result_type operator()(const reco::GsfElectronPtr&) const final;

  void setConsumes(edm::ConsumesCollector&) final;
  void getEventContent(const edm::EventBase&) final;

  double value(const reco::CandidatePtr& cand) const final;

  CandidateType candidateType() const final { 
    return ELECTRON; 
  }

private:
  const float barrelC0_, barrelCE_, barrelCr_, endcapC0_, endcapCE_, endcapCr_, barrelCutOff_;  
  edm::Handle<double> rhoHandle_;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleHadronicOverEMEnergyScaledCut,
		  "GsfEleHadronicOverEMEnergyScaledCut");

void GsfEleHadronicOverEMEnergyScaledCut::setConsumes(edm::ConsumesCollector& cc) {
  auto rho = cc.consumes<double>(contentTags_["rho"]);
  contentTokens_.emplace("rho",rho);
}

void GsfEleHadronicOverEMEnergyScaledCut::getEventContent(const edm::EventBase& ev) {  
  ev.getByLabel(contentTags_["rho"],rhoHandle_);
}


CutApplicatorBase::result_type GsfEleHadronicOverEMEnergyScaledCut::operator()(const reco::GsfElectronPtr& cand) const { 

  const double rho = rhoHandle_.isValid() ? (float)(*rhoHandle_) : 0;
  const float energy = cand->superCluster()->energy();
  const float c0 = (std::abs(cand->superCluster()->position().eta()) < barrelCutOff_ ? barrelC0_ : endcapC0_);
  const float cE = (std::abs(cand->superCluster()->position().eta()) < barrelCutOff_ ? barrelCE_ : endcapCE_);
  const float cR = (std::abs(cand->superCluster()->position().eta()) < barrelCutOff_ ? barrelCr_ : endcapCr_);
  return cand->hadronicOverEm() < c0 + cE/energy + cR*rho/energy;
}

double GsfEleHadronicOverEMEnergyScaledCut::value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);  
  return ele->hadronicOverEm();
}
