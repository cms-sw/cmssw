#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

#include "RecoEgamma/ElectronIdentification/interface/EBEECutValues.h"

class GsfEleFull5x5E2x5OverE5x5Cut : public CutApplicatorWithEventContentBase {
public:
  GsfEleFull5x5E2x5OverE5x5Cut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronPtr&) const override final;

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
  
  constexpr static char e5x5_[] = "e5x5";
  constexpr static char e2x5_[] = "e2x5";
  constexpr static char e1x5_[] = "e1x5";
};

constexpr char GsfEleFull5x5E2x5OverE5x5Cut::e5x5_[];
constexpr char GsfEleFull5x5E2x5OverE5x5Cut::e2x5_[];
constexpr char GsfEleFull5x5E2x5OverE5x5Cut::e1x5_[];

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleFull5x5E2x5OverE5x5Cut,
		  "GsfEleFull5x5E2x5OverE5x5Cut");

GsfEleFull5x5E2x5OverE5x5Cut::GsfEleFull5x5E2x5OverE5x5Cut(const edm::ParameterSet& params) :
  CutApplicatorWithEventContentBase(params),
  minE1x5OverE5x5Cut_(params,"minE1x5OverE5x5"),
  minE2x5OverE5x5Cut_(params,"minE2x5OverE5x5"){ 
  edm::InputTag e5x5Tag = params.getParameter<edm::InputTag>(e5x5_);
  edm::InputTag e2x5Tag = params.getParameter<edm::InputTag>(e2x5_);
  edm::InputTag e1x5Tag = params.getParameter<edm::InputTag>(e1x5_);
  contentTags_.emplace(e5x5_,e5x5Tag);
  contentTags_.emplace(e2x5_,e2x5Tag);
  contentTags_.emplace(e1x5_,e1x5Tag); 
}

void GsfEleFull5x5E2x5OverE5x5Cut::setConsumes(edm::ConsumesCollector& cc) {
  auto e5x5 = cc.consumes<edm::ValueMap<float> >(contentTags_[e5x5_]);
  contentTokens_.emplace(e5x5_,e5x5); 
  auto e2x5 = cc.consumes<edm::ValueMap<float> >(contentTags_[e2x5_]);
  contentTokens_.emplace(e2x5_,e2x5); 
  auto e1x5 = cc.consumes<edm::ValueMap<float> >(contentTags_[e1x5_]);
  contentTokens_.emplace(e1x5_,e1x5);  
}

void GsfEleFull5x5E2x5OverE5x5Cut::getEventContent(const edm::EventBase& ev) {  
  ev.getByLabel(contentTags_[e5x5_],e5x5Handle_);
  ev.getByLabel(contentTags_[e2x5_],e2x5Handle_);  
  ev.getByLabel(contentTags_[e1x5_],e1x5Handle_);  
}

CutApplicatorBase::result_type 
GsfEleFull5x5E2x5OverE5x5Cut::
operator()(const reco::GsfElectronPtr& cand) const{  

  const double e5x5 = (*e5x5Handle_)[cand];
  const double e1x5OverE5x5 = e5x5!=0 ? (*e1x5Handle_)[cand]/e5x5 : 0; 
  const double e2x5OverE5x5 = e5x5!=0 ? (*e2x5Handle_)[cand]/e5x5 : 0;

  return e1x5OverE5x5 > minE1x5OverE5x5Cut_(cand) || e2x5OverE5x5 > minE2x5OverE5x5Cut_(cand);
 
}
