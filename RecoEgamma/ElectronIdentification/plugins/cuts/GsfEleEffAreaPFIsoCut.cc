#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEgamma/EgammaTools/interface/EffectiveAreas.h"


class GsfEleEffAreaPFIsoCut : public CutApplicatorWithEventContentBase {
public:
  GsfEleEffAreaPFIsoCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronPtr&) const final;

  void setConsumes(edm::ConsumesCollector&) final;
  void getEventContent(const edm::EventBase&) final;

  double value(const reco::CandidatePtr& cand) const final;

  CandidateType candidateType() const final { 
    return ELECTRON; 
  }

private:
  // Cut values
  const float _isoCutEBLowPt,_isoCutEBHighPt,_isoCutEELowPt,_isoCutEEHighPt;
  // Configuration
  const float _ptCutOff;
  const float _barrelCutOff;
  bool  _isRelativeIso;
  // Effective area constants
  EffectiveAreas _effectiveAreas;
  // The rho
  edm::Handle< double > _rhoHandle;

  constexpr static char rhoString_     [] = "rho";
};

constexpr char GsfEleEffAreaPFIsoCut::rhoString_[];

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleEffAreaPFIsoCut,
		  "GsfEleEffAreaPFIsoCut");

GsfEleEffAreaPFIsoCut::GsfEleEffAreaPFIsoCut(const edm::ParameterSet& c) :
  CutApplicatorWithEventContentBase(c),
  _isoCutEBLowPt(c.getParameter<double>("isoCutEBLowPt")),
  _isoCutEBHighPt(c.getParameter<double>("isoCutEBHighPt")),
  _isoCutEELowPt(c.getParameter<double>("isoCutEELowPt")),
  _isoCutEEHighPt(c.getParameter<double>("isoCutEEHighPt")),
  _ptCutOff(c.getParameter<double>("ptCutOff")),
   _barrelCutOff(c.getParameter<double>("barrelCutOff")),
  _isRelativeIso(c.getParameter<bool>("isRelativeIso")),
  _effectiveAreas( (c.getParameter<edm::FileInPath>("effAreasConfigFile")).fullPath())
{
  
  edm::InputTag rhoTag = c.getParameter<edm::InputTag>("rho");
  contentTags_.emplace(rhoString_,rhoTag);

}

void GsfEleEffAreaPFIsoCut::setConsumes(edm::ConsumesCollector& cc) {

  auto rho = cc.consumes<double>(contentTags_[rhoString_]);
  contentTokens_.emplace(rhoString_, rho);
}

void GsfEleEffAreaPFIsoCut::getEventContent(const edm::EventBase& ev) {  

  ev.getByLabel(contentTags_[rhoString_],_rhoHandle);
}

CutApplicatorBase::result_type 
GsfEleEffAreaPFIsoCut::
operator()(const reco::GsfElectronPtr& cand) const{  

  // Establish the cut value
  double absEta = std::abs(cand->superCluster()->eta());
  const float isoCut =
    ( cand->pt() < _ptCutOff ?
      ( absEta < _barrelCutOff ? _isoCutEBLowPt : _isoCutEELowPt ) 
      :
      ( absEta < _barrelCutOff ? _isoCutEBHighPt : _isoCutEEHighPt ) );

  // Compute the combined isolation with effective area correction
  const reco::GsfElectron::PflowIsolationVariables& pfIso =
    cand->pfIsolationVariables();
  const float chad = pfIso.sumChargedHadronPt;
  const float nhad = pfIso.sumNeutralHadronEt;
  const float pho = pfIso.sumPhotonEt;
  const float  eA = _effectiveAreas.getEffectiveArea( absEta );
  const float rho = _rhoHandle.isValid() ? (float)(*_rhoHandle) : 0; // std::max likes float arguments
  const float iso = chad + std::max(0.0f, nhad + pho - rho*eA);
  
  // Apply the cut and return the result
  // Scale by pT if the relative isolation is requested but avoid division by 0
  return iso < isoCut*(_isRelativeIso ? cand->pt() : 1.);
}

double GsfEleEffAreaPFIsoCut::value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);
  // Establish the cut value
  double absEta = std::abs(ele->superCluster()->eta());
  
  // Compute the combined isolation with effective area correction
  const reco::GsfElectron::PflowIsolationVariables& pfIso =
    ele->pfIsolationVariables();
  const float chad = pfIso.sumChargedHadronPt;
  const float nhad = pfIso.sumNeutralHadronEt;
  const float pho = pfIso.sumPhotonEt;
  float  eA = _effectiveAreas.getEffectiveArea( absEta );
  float rho = (float)(*_rhoHandle); // std::max likes float arguments
  float iso = chad + std::max(0.0f, nhad + pho - rho*eA);
  
  // Divide by pT if the relative isolation is requested
  if( _isRelativeIso )
    iso /= ele->pt();

  // Apply the cut and return the result
  return iso;
}
