#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class GsfEleHadronicOverEMEnergyScaledCut : public CutApplicatorWithEventContentBase {
public:
  GsfEleHadronicOverEMEnergyScaledCut(const edm::ParameterSet& c) :
    CutApplicatorWithEventContentBase(c),
    _barrelC0(c.getParameter<double>("barrelC0")),
    _barrelCE(c.getParameter<double>("barrelCE")),
    _barrelCr(c.getParameter<double>("barrelCr")),
    _endcapC0(c.getParameter<double>("endcapC0")),
    _endcapCE(c.getParameter<double>("endcapCE")),
    _endcapCr(c.getParameter<double>("endcapCr")),
    _barrelCutOff(c.getParameter<double>("barrelCutOff")) 
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
  const float _barrelC0, _barrelCE, _barrelCr, _endcapC0, _endcapCE, _endcapCr, _barrelCutOff;  
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
  const float C0 = (std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ? _barrelC0 : _endcapC0);
  const float CE = (std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ? _barrelCE : _endcapCE);
  const float Cr = (std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ? _barrelCr : _endcapCr);
  return cand->hadronicOverEm() < C0 + CE/energy + Cr*rho/energy;
}

double GsfEleHadronicOverEMEnergyScaledCut::value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);  
  return ele->hadronicOverEm();
}
