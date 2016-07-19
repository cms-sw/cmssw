#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEgamma/EgammaTools/interface/EffectiveAreas.h"


class GsfEleCalPFClusterIsoCut : public CutApplicatorWithEventContentBase {
public:

  enum IsoType {UNDEF=-1, ISO_ECAL=0, ISO_HCAL=1};

  GsfEleCalPFClusterIsoCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronPtr&) const override final;

  void setConsumes(edm::ConsumesCollector&) override final;
  void getEventContent(const edm::EventBase&) override final;

  double value(const reco::CandidatePtr& cand) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  // Cut values
  const float _isoCutEBLowPt,_isoCutEBHighPt,_isoCutEELowPt,_isoCutEEHighPt;
  // Configuration
  const int     _isoType;
  const float   _ptCutOff;
  const float   _barrelCutOff;
  bool          _isRelativeIso;
  // Effective area constants
  EffectiveAreas _effectiveAreas;
  // The rho
  edm::Handle< double > _rhoHandle;

  constexpr static char rhoString_     [] = "rho";
};

constexpr char GsfEleCalPFClusterIsoCut::rhoString_[];

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleCalPFClusterIsoCut,
		  "GsfEleCalPFClusterIsoCut");

GsfEleCalPFClusterIsoCut::GsfEleCalPFClusterIsoCut(const edm::ParameterSet& c) :
  CutApplicatorWithEventContentBase(c),
  _isoCutEBLowPt(c.getParameter<double>("isoCutEBLowPt")),
  _isoCutEBHighPt(c.getParameter<double>("isoCutEBHighPt")),
  _isoCutEELowPt(c.getParameter<double>("isoCutEELowPt")),
  _isoCutEEHighPt(c.getParameter<double>("isoCutEEHighPt")),
  _isoType(c.getParameter<int>("isoType")),
  _ptCutOff(c.getParameter<double>("ptCutOff")),
   _barrelCutOff(c.getParameter<double>("barrelCutOff")),
  _isRelativeIso(c.getParameter<bool>("isRelativeIso")),
  _effectiveAreas( (c.getParameter<edm::FileInPath>("effAreasConfigFile")).fullPath())
{
  
  edm::InputTag rhoTag = c.getParameter<edm::InputTag>("rho");
  contentTags_.emplace(rhoString_,rhoTag);

}

void GsfEleCalPFClusterIsoCut::setConsumes(edm::ConsumesCollector& cc) {

  auto rho = cc.consumes<double>(contentTags_[rhoString_]);
  contentTokens_.emplace(rhoString_, rho);
}

void GsfEleCalPFClusterIsoCut::getEventContent(const edm::EventBase& ev) {  

  ev.getByLabel(contentTags_[rhoString_],_rhoHandle);
}

CutApplicatorBase::result_type 
GsfEleCalPFClusterIsoCut::
operator()(const reco::GsfElectronPtr& cand) const{  

  // Establish the cut value
  double absEta = std::abs(cand->superCluster()->eta());

  const pat::Electron* elPat = dynamic_cast<const pat::Electron*>(cand.get());
  if( !elPat ) {
    throw cms::Exception("ERROR: this VID selection is meant to be run on miniAOD/PAT only")
      << std::endl << "Change input format to PAT/miniAOD or contact Egamma experts" 
      << std::endl << std::endl;
  }

  const float isoCut =
    ( cand->pt() < _ptCutOff ?
      ( absEta < _barrelCutOff ? _isoCutEBLowPt : _isoCutEELowPt ) 
      :
      ( absEta < _barrelCutOff ? _isoCutEBHighPt : _isoCutEEHighPt ) );

  const float  eA = _effectiveAreas.getEffectiveArea( absEta );
  const float rho = _rhoHandle.isValid() ? (float)(*_rhoHandle) : 0; // std::max likes float arguments

  float isoValue = -999;
  if( _isoType == ISO_ECAL ){
    isoValue = elPat->ecalPFClusterIso();
  }else if( _isoType == ISO_HCAL ){
    isoValue = elPat->hcalPFClusterIso();
  }else{
    throw cms::Exception("ERROR: unknown type requested for PF cluster isolation.")
      << std::endl << "Check VID configuration." << std::endl;
  }
  float isoValueCorr = std::max(0.0f, isoValue - rho*eA);
  
  // Apply the cut and return the result
  // Scale by pT if the relative isolation is requested but avoid division by 0
  return isoValueCorr < isoCut*(_isRelativeIso ? cand->pt() : 1.);
}

double GsfEleCalPFClusterIsoCut::value(const reco::CandidatePtr& cand) const {

  reco::GsfElectronPtr ele(cand);
  // Establish the cut value
  double absEta = std::abs(ele->superCluster()->eta());

  const pat::Electron *elPat = dynamic_cast<const pat::Electron*>(ele.get());
  if( !elPat ){
    throw cms::Exception("ERROR: this VID selection is meant to be run on miniAOD/PAT only")
      << std::endl << "Change input format to PAT/miniAOD or contact Egamma experts" 
      << std::endl << std::endl;
  }

  const float  eA = _effectiveAreas.getEffectiveArea( absEta );
  const float rho = _rhoHandle.isValid() ? (float)(*_rhoHandle) : 0; // std::max likes float arguments

  float isoValue = -999;
  if( _isoType == ISO_ECAL ){
    isoValue = elPat->ecalPFClusterIso();
  }else if( _isoType == ISO_HCAL ){
    isoValue = elPat->hcalPFClusterIso();
  }else{
    throw cms::Exception("ERROR: unknown type requested for PF cluster isolation.")
      << std::endl << "Check VID configuration." << std::endl;
  } 
  float isoValueCorr = std::max(0.0f, isoValue - rho*eA);
  
  // Divide by pT if the relative isolation is requested
  if( _isRelativeIso )
    isoValueCorr /= ele->pt();

  // Apply the cut and return the result
  return isoValueCorr;
}
