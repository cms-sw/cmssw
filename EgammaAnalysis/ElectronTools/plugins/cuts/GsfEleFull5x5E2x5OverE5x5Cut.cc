#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

#include "EgammaAnalysis/ElectronTools/interface/EBEECutValues.h"

class GsfEleFull5x5E2x5OverE5x5Cut : public CutApplicatorWithEventContentBase {
public:
  GsfEleFull5x5E2x5OverE5x5Cut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronRef&) const override final;

  void setConsumes(edm::ConsumesCollector&) override final;
  void getEventContent(const edm::EventBase&) override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  EBEECutValues minE1x5OverE5x5Cut_;
  EBEECutValues minE2x5OverE5x5Cut_;
  
  
  edm::Handle<edm::ValueMap<float> > e1x5Handle_;  
  edm::Handle<edm::ValueMap<float> > e2x5Handle_;  
  edm::Handle<edm::ValueMap<float> > e5x5Handle_;
  
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleFull5x5E2x5OverE5x5Cut,
		  "GsfEleFull5x5E2x5OverE5x5Cut");

GsfEleFull5x5E2x5OverE5x5Cut::GsfEleFull5x5E2x5OverE5x5Cut(const edm::ParameterSet& params) :
  CutApplicatorWithEventContentBase(params),
  minE1x5OverE5x5Cut_(params,"minE1x5OverE5x5Cut"),
  minE2x5OverE5x5Cut_(params,"minE2x5OverE5x5Cut"){ 
  edm::InputTag e5x5Tag = params.getParameter<edm::InputTag>("e5x5");
  edm::InputTag e2x5Tag = params.getParameter<edm::InputTag>("e2x5");
  edm::InputTag e1x5Tag = params.getParameter<edm::InputTag>("e1x5");
  contentTags_.emplace("e1x5",e1x5Tag);
  contentTags_.emplace("e2x5",e2x5Tag);
  contentTags_.emplace("e5x5",e5x5Tag);
 
}

void GsfEleFull5x5E2x5OverE5x5Cut::setConsumes(edm::ConsumesCollector& cc) {
  auto e1x5 = cc.consumes<double>(contentTags_["e1x5"]);
  contentTokens_.emplace("e1x5",e1x5);
  auto e2x5 = cc.consumes<double>(contentTags_["e2x5"]);
  contentTokens_.emplace("e2x5",e2x5); 
  auto e5x5 = cc.consumes<double>(contentTags_["e5x5"]);
  contentTokens_.emplace("e5x5",e5x5); 
}

void GsfEleFull5x5E2x5OverE5x5Cut::getEventContent(const edm::EventBase& ev) {  
  ev.getByLabel(contentTags_["e1x5"],e1x5Handle_);
  ev.getByLabel(contentTags_["e2x5"],e2x5Handle_);  
  ev.getByLabel(contentTags_["e5x5"],e5x5Handle_);
}

CutApplicatorBase::result_type 
GsfEleFull5x5E2x5OverE5x5Cut::
operator()(const reco::GsfElectronRef& cand) const{  

  const double e5x5 = (*e5x5Handle_)[cand];
  const double e1x5OverE5x5 = e5x5!=0 ? (*e1x5Handle_)[cand]/e5x5 : 0; 
  const double e2x5OverE5x5 = e5x5!=0 ? (*e2x5Handle_)[cand]/e5x5 : 0;

  return e1x5OverE5x5 > minE1x5OverE5x5Cut_(cand) || e2x5OverE5x5 > minE2x5OverE5x5Cut_(cand);
 
}
