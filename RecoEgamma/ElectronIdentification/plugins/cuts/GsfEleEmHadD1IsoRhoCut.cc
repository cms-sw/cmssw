#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

#include "RecoEgamma/ElectronIdentification/interface/EBEECutValues.h"

class GsfEleEmHadD1IsoRhoCut : public CutApplicatorWithEventContentBase {
public:
  GsfEleEmHadD1IsoRhoCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronPtr&) const override final;

  void setConsumes(edm::ConsumesCollector&) override final;
  void getEventContent(const edm::EventBase&) override final;

  double value(const reco::CandidatePtr& cand) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  float rhoConstant_;
  EBEECutValues slopeTerm_;
  EBEECutValues slopeStart_;
  EBEECutValues constTerm_;
  
  
  edm::Handle<double> rhoHandle_;
  
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleEmHadD1IsoRhoCut,
		  "GsfEleEmHadD1IsoRhoCut");

GsfEleEmHadD1IsoRhoCut::GsfEleEmHadD1IsoRhoCut(const edm::ParameterSet& params) :
  CutApplicatorWithEventContentBase(params),
  rhoConstant_(params.getParameter<double>("rhoConstant")),
  slopeTerm_(params,"slopeTerm"),
  slopeStart_(params,"slopeStart"),
  constTerm_(params,"constTerm"){
  edm::InputTag rhoTag = params.getParameter<edm::InputTag>("rho");
  contentTags_.emplace("rho",rhoTag);
 
}

void GsfEleEmHadD1IsoRhoCut::setConsumes(edm::ConsumesCollector& cc) {
  auto rho = cc.consumes<double>(contentTags_["rho"]);
  contentTokens_.emplace("rho",rho);
}

void GsfEleEmHadD1IsoRhoCut::getEventContent(const edm::EventBase& ev) {  
  ev.getByLabel(contentTags_["rho"],rhoHandle_);
}

CutApplicatorBase::result_type 
GsfEleEmHadD1IsoRhoCut::
operator()(const reco::GsfElectronPtr& cand) const{  
  const double rho = (*rhoHandle_);
  
  const float isolEmHadDepth1 = cand->dr03EcalRecHitSumEt() + cand->dr03HcalDepth1TowerSumEt(); 

  const float et = cand->et();
  const float cutValue = et > slopeStart_(cand)  ? slopeTerm_(cand)*(et-slopeStart_(cand)) + constTerm_(cand) : constTerm_(cand);
  return isolEmHadDepth1 < cutValue + rhoConstant_*rho;
}

double GsfEleEmHadD1IsoRhoCut::value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);
  return ele->dr03EcalRecHitSumEt() + ele->dr03HcalDepth1TowerSumEt();
}
