#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "RecoEgamma/ElectronIdentification/interface/EBEECutValues.h"

class GsfEleValueMapIsoRhoCut : public CutApplicatorWithEventContentBase {
public:
  GsfEleValueMapIsoRhoCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronPtr&) const final;
 
  void setConsumes(edm::ConsumesCollector&) final;
  void getEventContent(const edm::EventBase&) final;
  
  double value(const reco::CandidatePtr& cand) const final;

  CandidateType candidateType() const final { 
    return ELECTRON; 
  }

private:
  float getRhoCorr(const reco::GsfElectronPtr& cand)const;
   
  EBEECutValues slopeTerm_;
  EBEECutValues slopeStart_;
  EBEECutValues constTerm_;
  EBEECutValues rhoEtStart_;
  EBEECutValues rhoEA_;
  
  bool useRho_;
  
  edm::Handle<double> rhoHandle_;
  edm::Handle<edm::ValueMap<float> > valueHandle_;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleValueMapIsoRhoCut,
		  "GsfEleValueMapIsoRhoCut");

GsfEleValueMapIsoRhoCut::GsfEleValueMapIsoRhoCut(const edm::ParameterSet& params) :
  CutApplicatorWithEventContentBase(params),
  slopeTerm_(params,"slopeTerm"),
  slopeStart_(params,"slopeStart"),
  constTerm_(params,"constTerm"),
  rhoEtStart_(params,"rhoEtStart"),
  rhoEA_(params,"rhoEA")
{
  auto rho = params.getParameter<edm::InputTag>("rho");
  if(!rho.label().empty()){
    useRho_=true;
    contentTags_.emplace("rho",rho);
  }else useRho_=false;
  
  contentTags_.emplace("value",params.getParameter<edm::InputTag>("value"));

}

void GsfEleValueMapIsoRhoCut::setConsumes(edm::ConsumesCollector& cc) {
  if(useRho_) contentTokens_.emplace("rho",cc.consumes<double>(contentTags_["rho"]));
  contentTokens_.emplace("value",cc.consumes<edm::ValueMap<float> >(contentTags_["value"]));
}

void GsfEleValueMapIsoRhoCut::getEventContent(const edm::EventBase& ev) {  
  if(useRho_) ev.getByLabel(contentTags_["rho"],rhoHandle_);
  ev.getByLabel(contentTags_["value"],valueHandle_);
}

CutApplicatorBase::result_type 
GsfEleValueMapIsoRhoCut::
operator()(const reco::GsfElectronPtr& cand) const{  

  const float val = (*valueHandle_)[cand];

  const float et = cand->et();
  const float cutValue = et > slopeStart_(cand)  ? slopeTerm_(cand)*(et-slopeStart_(cand)) + constTerm_(cand) : constTerm_(cand);
  const float rhoCutValue = getRhoCorr(cand);

  return val < cutValue+rhoCutValue;
}


double GsfEleValueMapIsoRhoCut::
value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);  
  return (*valueHandle_)[ele];
}

float GsfEleValueMapIsoRhoCut::getRhoCorr(const reco::GsfElectronPtr& cand)const{
  if(!useRho_) return 0.;
  else{
    const double rho = (*rhoHandle_);
    return cand->et() >= rhoEtStart_(cand) ? rhoEA_(cand)*rho : 0.;
  }
}
