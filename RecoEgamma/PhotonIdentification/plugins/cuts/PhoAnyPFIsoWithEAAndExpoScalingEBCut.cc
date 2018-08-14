#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "RecoEgamma/EgammaTools/interface/EffectiveAreas.h"


class PhoAnyPFIsoWithEAAndExpoScalingEBCut : public CutApplicatorWithEventContentBase {
public:
  PhoAnyPFIsoWithEAAndExpoScalingEBCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::PhotonPtr&) const final;

  void setConsumes(edm::ConsumesCollector&) final;
  void getEventContent(const edm::EventBase&) final;

  double value(const reco::CandidatePtr& cand) const final;
  
  CandidateType candidateType() const final { 
    return PHOTON; 
  }

private:
  // Cut values
  float _C1_EB;
  float _C2_EB;
  float _C3_EB;
  float _C1_EE;
  float _C2_EE;
  // Configuration
  float _barrelCutOff;
  bool  _useRelativeIso;
  // Effective area constants
  EffectiveAreas _effectiveAreas;
  // The isolations computed upstream
  edm::Handle<edm::ValueMap<float> > _anyPFIsoMap;
  // The rho
  edm::Handle< double > _rhoHandle;

  constexpr static char anyPFIsoWithEA_[] = "anyPFIsoWithEA";
  constexpr static char rhoString_     [] = "rho";
};

constexpr char PhoAnyPFIsoWithEAAndExpoScalingEBCut::anyPFIsoWithEA_[];
constexpr char PhoAnyPFIsoWithEAAndExpoScalingEBCut::rhoString_[];

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  PhoAnyPFIsoWithEAAndExpoScalingEBCut,
		  "PhoAnyPFIsoWithEAAndExpoScalingEBCut");

PhoAnyPFIsoWithEAAndExpoScalingEBCut::PhoAnyPFIsoWithEAAndExpoScalingEBCut(const edm::ParameterSet& c) :
  CutApplicatorWithEventContentBase(c),
  _C1_EB(c.getParameter<double>("C1_EB")),
  _C2_EB(c.getParameter<double>("C2_EB")),
  _C3_EB(c.getParameter<double>("C3_EB")),
  _C1_EE(c.getParameter<double>("C1_EE")),
  _C2_EE(c.getParameter<double>("C2_EE")),
  _barrelCutOff(c.getParameter<double>("barrelCutOff")),
  _useRelativeIso(c.getParameter<bool>("useRelativeIso")),
  _effectiveAreas( (c.getParameter<edm::FileInPath>("effAreasConfigFile")).fullPath())
{
  
  edm::InputTag maptag = c.getParameter<edm::InputTag>("anyPFIsoMap");
  contentTags_.emplace(anyPFIsoWithEA_,maptag);

  edm::InputTag rhoTag = c.getParameter<edm::InputTag>("rho");
  contentTags_.emplace(rhoString_,rhoTag);

}

void PhoAnyPFIsoWithEAAndExpoScalingEBCut::setConsumes(edm::ConsumesCollector& cc) {
  auto anyPFIsoWithEA = 
    cc.consumes<edm::ValueMap<float> >(contentTags_[anyPFIsoWithEA_]);
  contentTokens_.emplace(anyPFIsoWithEA_,anyPFIsoWithEA);

  auto rho = cc.consumes<double>(contentTags_[rhoString_]);
  contentTokens_.emplace(rhoString_, rho);
}

void PhoAnyPFIsoWithEAAndExpoScalingEBCut::getEventContent(const edm::EventBase& ev) {  
  ev.getByLabel(contentTags_[anyPFIsoWithEA_],_anyPFIsoMap);
  ev.getByLabel(contentTags_[rhoString_],_rhoHandle);
}

CutApplicatorBase::result_type 
PhoAnyPFIsoWithEAAndExpoScalingEBCut::
operator()(const reco::PhotonPtr& cand) const{  

  // in case we are by-value
  const std::string& inst_name = contentTags_.find(anyPFIsoWithEA_)->second.instance();
  edm::Ptr<pat::Photon> pat(cand);
  float anyisoval = -1.0;
  if( _anyPFIsoMap.isValid() && _anyPFIsoMap->contains( cand.id() ) ) {
    anyisoval = (*_anyPFIsoMap)[cand];
  } else if ( _anyPFIsoMap.isValid() && _anyPFIsoMap->idSize() == 1 &&
              cand.id() == edm::ProductID() ) {
    // in case we have spoofed a ptr
    //note this must be a 1:1 valuemap (only one product input)
    anyisoval = _anyPFIsoMap->begin()[cand.key()];
  } else if ( _anyPFIsoMap.isValid() ){ // throw an exception
    anyisoval = (*_anyPFIsoMap)[cand];
  }

  // Figure out the cut value
  // The value is generally pt-dependent: C1 + pt * C2
  const float pt = cand->pt();

  // In this version of the isolation cut we apply
  // exponential pt scaling to the barrel isolation cut,
  // and linear pt scaling to the endcap isolation cut.
  double absEta = std::abs(cand->superCluster()->eta());
  const float isolationCutValue = 
    ( absEta < _barrelCutOff ? 
      _C1_EB + exp( pt*_C2_EB + _C3_EB)
      : _C1_EE + pt * _C2_EE);
  
  // Retrieve the variable value for this particle
  float anyPFIso = _anyPFIsoMap.isValid() ? anyisoval : pat->userFloat(inst_name);

  // Apply pile-up correction
  double eA = _effectiveAreas.getEffectiveArea( absEta );
  double rho = *_rhoHandle;
  float anyPFIsoWithEA = std::max(0.0, anyPFIso - rho * eA);

  // Divide by pT if the relative isolation is requested
  if( _useRelativeIso )
    anyPFIsoWithEA /= pt;

  // Apply the cut and return the result
  return anyPFIsoWithEA < isolationCutValue;
}

double PhoAnyPFIsoWithEAAndExpoScalingEBCut::
value(const reco::CandidatePtr& cand) const {
  reco::PhotonPtr pho(cand);

  // in case we are by-value
  const std::string& inst_name = contentTags_.find(anyPFIsoWithEA_)->second.instance();
  edm::Ptr<pat::Photon> pat(cand);
  float anyisoval = -1.0;
  if( _anyPFIsoMap.isValid() && _anyPFIsoMap->contains( cand.id() ) ) {
    anyisoval = (*_anyPFIsoMap)[cand];
  } else if ( _anyPFIsoMap.isValid() && _anyPFIsoMap->idSize() == 1 &&
              cand.id() == edm::ProductID() ) {
    // in case we have spoofed a ptr
    //note this must be a 1:1 valuemap (only one product input)
    anyisoval = _anyPFIsoMap->begin()[cand.key()];
  } else if ( _anyPFIsoMap.isValid() ){ // throw an exception
    anyisoval = (*_anyPFIsoMap)[cand];
  }

  // Figure out the cut value
  // The value is generally pt-dependent: C1 + pt * C2
  const float pt = pho->pt();

  // In this version of the isolation cut we apply
  // exponential pt scaling to the barrel isolation cut,
  // and linear pt scaling to the endcap isolation cut.
  double absEta = std::abs(pho->superCluster()->eta());
  
  // Retrieve the variable value for this particle
  float anyPFIso = _anyPFIsoMap.isValid() ? anyisoval : pat->userFloat(inst_name);

  // Apply pile-up correction
  double eA = _effectiveAreas.getEffectiveArea( absEta );
  double rho = *_rhoHandle;
  float anyPFIsoWithEA = std::max(0.0, anyPFIso - rho * eA);

  // Divide by pT if the relative isolation is requested
  if( _useRelativeIso )
    anyPFIsoWithEA /= pt;

  // Apply the cut and return the result
  return anyPFIsoWithEA;
}
