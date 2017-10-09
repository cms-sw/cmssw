#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "RecoEgamma/ElectronIdentification/interface/EBEECutValues.h"

class GsfEleFull5x5SigmaIEtaIEtaWithSatCut : public CutApplicatorWithEventContentBase {
public:
  GsfEleFull5x5SigmaIEtaIEtaWithSatCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronPtr&) const override final;

  void setConsumes(edm::ConsumesCollector&) override final;
  void getEventContent(const edm::EventBase&) override final;
  
  double value(const reco::CandidatePtr& cand) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  EBEECutValues maxSigmaIEtaIEtaCut_;
  EBEECutValuesInt maxNrSatCrysIn5x5Cut_;
  edm::Handle<edm::ValueMap<int> > nrSatCrysValueMap_;
  
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleFull5x5SigmaIEtaIEtaWithSatCut,
		  "GsfEleFull5x5SigmaIEtaIEtaWithSatCut");

GsfEleFull5x5SigmaIEtaIEtaWithSatCut::GsfEleFull5x5SigmaIEtaIEtaWithSatCut(const edm::ParameterSet& params) :
  CutApplicatorWithEventContentBase(params),
  maxSigmaIEtaIEtaCut_(params,"maxSigmaIEtaIEta"),

  maxNrSatCrysIn5x5Cut_(params,"maxNrSatCrysIn5x5"){
  contentTags_.emplace("nrSatCrysValueMap",params.getParameter<edm::InputTag>("nrSatCrysValueMap"));
}

void GsfEleFull5x5SigmaIEtaIEtaWithSatCut::setConsumes(edm::ConsumesCollector& cc) {
  contentTokens_.emplace("nrSatCrysValueMap",cc.consumes<edm::ValueMap<int> >(contentTags_["nrSatCrysValueMap"]));
}

void GsfEleFull5x5SigmaIEtaIEtaWithSatCut::getEventContent(const edm::EventBase& ev) {  
  ev.getByLabel(contentTags_["nrSatCrysValueMap"],nrSatCrysValueMap_);

}


CutApplicatorBase::result_type 
GsfEleFull5x5SigmaIEtaIEtaWithSatCut::
operator()(const reco::GsfElectronPtr& cand) const{  

  if((*nrSatCrysValueMap_)[cand]>maxNrSatCrysIn5x5Cut_(cand)) return true;
  else return cand->full5x5_sigmaIetaIeta() < maxSigmaIEtaIEtaCut_(cand);
 
}

double GsfEleFull5x5SigmaIEtaIEtaWithSatCut::
value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);  
  return ele->full5x5_sigmaIetaIeta();
}
