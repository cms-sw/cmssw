#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "RecoEgamma/ElectronIdentification/interface/EBEECutValues.h"

class GsfEleFull5x5E2x5OverE5x5WithSatCut : public CutApplicatorWithEventContentBase {
public:
  GsfEleFull5x5E2x5OverE5x5WithSatCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronPtr&) const override final;

  void setConsumes(edm::ConsumesCollector&) override final;
  void getEventContent(const edm::EventBase&) override final;
  
  double value(const reco::CandidatePtr& cand) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  EBEECutValues minE1x5OverE5x5Cut_;
  EBEECutValues minE2x5OverE5x5Cut_;
  EBEECutValuesInt maxNrSatCrysIn5x5Cut_;
  edm::Handle<edm::ValueMap<int> > nrSatCrysValueMap_;
  
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleFull5x5E2x5OverE5x5WithSatCut,
		  "GsfEleFull5x5E2x5OverE5x5WithSatCut");

GsfEleFull5x5E2x5OverE5x5WithSatCut::GsfEleFull5x5E2x5OverE5x5WithSatCut(const edm::ParameterSet& params) :
  CutApplicatorWithEventContentBase(params),
  minE1x5OverE5x5Cut_(params,"minE1x5OverE5x5"),
  minE2x5OverE5x5Cut_(params,"minE2x5OverE5x5"),
  maxNrSatCrysIn5x5Cut_(params,"maxNrSatCrysIn5x5"){
  contentTags_.emplace("nrSatCrysValueMap",params.getParameter<edm::InputTag>("nrSatCrysValueMap"));
}

void GsfEleFull5x5E2x5OverE5x5WithSatCut::setConsumes(edm::ConsumesCollector& cc) {
  contentTokens_.emplace("nrSatCrysValueMap",cc.consumes<edm::ValueMap<int> >(contentTags_["nrSatCrysValueMap"]));
}

void GsfEleFull5x5E2x5OverE5x5WithSatCut::getEventContent(const edm::EventBase& ev) {  
  ev.getByLabel(contentTags_["nrSatCrysValueMap"],nrSatCrysValueMap_);

}


CutApplicatorBase::result_type 
GsfEleFull5x5E2x5OverE5x5WithSatCut::
operator()(const reco::GsfElectronPtr& cand) const{  

  if((*nrSatCrysValueMap_)[cand]>maxNrSatCrysIn5x5Cut_(cand)) return true;

  const double e5x5 = cand->full5x5_e5x5();
  const double e2x5OverE5x5 = e5x5!=0 ? cand->full5x5_e2x5Max()/e5x5 : 0; 
  const double e1x5OverE5x5 = e5x5!=0 ? cand->full5x5_e1x5()/e5x5 : 0;

  return e1x5OverE5x5 > minE1x5OverE5x5Cut_(cand) || e2x5OverE5x5 > minE2x5OverE5x5Cut_(cand);
 
}

double GsfEleFull5x5E2x5OverE5x5WithSatCut::
value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);  
  //btw we broke somebodies nice model of assuming every cut is 1D....
  //what this is returning is fairly meaningless...
  return ele->full5x5_e1x5() ? ele->full5x5_e2x5Max()/ele->full5x5_e1x5() : 0.;
}
