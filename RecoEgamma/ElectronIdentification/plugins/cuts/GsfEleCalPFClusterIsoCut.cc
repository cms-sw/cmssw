#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEgamma/EgammaTools/interface/EffectiveAreas.h"


class GsfEleCalPFClusterIsoCut : public CutApplicatorWithEventContentBase {
public:

  enum IsoType {UNDEF=-1, ISO_ECAL=0, ISO_HCAL=1};

  GsfEleCalPFClusterIsoCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronPtr&) const final;

  void setConsumes(edm::ConsumesCollector&) final;
  void getEventContent(const edm::EventBase&) final;

  double value(const reco::CandidatePtr& cand) const final;

  CandidateType candidateType() const final { 
    return ELECTRON; 
  }

private:
  // Cut values
  const float isoCutEBLowPt_,isoCutEBHighPt_,isoCutEELowPt_,isoCutEEHighPt_;
  // Configuration
  const int     isoType_;
  const float   ptCutOff_;
  const float   barrelCutOff_;
  bool          isRelativeIso_;
  // Effective area constants
  EffectiveAreas effectiveAreas_;
  // The rho
  edm::Handle< double > rhoHandle_;

  constexpr static char rhoString_     [] = "rho";
};

constexpr char GsfEleCalPFClusterIsoCut::rhoString_[];

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleCalPFClusterIsoCut,
		  "GsfEleCalPFClusterIsoCut");

GsfEleCalPFClusterIsoCut::GsfEleCalPFClusterIsoCut(const edm::ParameterSet& c) :
  CutApplicatorWithEventContentBase(c),
  isoCutEBLowPt_(c.getParameter<double>("isoCutEBLowPt")),
  isoCutEBHighPt_(c.getParameter<double>("isoCutEBHighPt")),
  isoCutEELowPt_(c.getParameter<double>("isoCutEELowPt")),
  isoCutEEHighPt_(c.getParameter<double>("isoCutEEHighPt")),
  isoType_(c.getParameter<int>("isoType")),
  ptCutOff_(c.getParameter<double>("ptCutOff")),
   barrelCutOff_(c.getParameter<double>("barrelCutOff")),
  isRelativeIso_(c.getParameter<bool>("isRelativeIso")),
  effectiveAreas_( (c.getParameter<edm::FileInPath>("effAreasConfigFile")).fullPath())
{
  
  edm::InputTag rhoTag = c.getParameter<edm::InputTag>("rho");
  contentTags_.emplace(rhoString_,rhoTag);

}

void GsfEleCalPFClusterIsoCut::setConsumes(edm::ConsumesCollector& cc) {

  auto rho = cc.consumes<double>(contentTags_[rhoString_]);
  contentTokens_.emplace(rhoString_, rho);
}

void GsfEleCalPFClusterIsoCut::getEventContent(const edm::EventBase& ev) {  

  ev.getByLabel(contentTags_[rhoString_],rhoHandle_);
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
    ( cand->pt() < ptCutOff_ ?
      ( absEta < barrelCutOff_ ? isoCutEBLowPt_ : isoCutEELowPt_ ) 
      :
      ( absEta < barrelCutOff_ ? isoCutEBHighPt_ : isoCutEEHighPt_ ) );

  const float  eA = effectiveAreas_.getEffectiveArea( absEta );
  const float rho = rhoHandle_.isValid() ? (float)(*rhoHandle_) : 0; // std::max likes float arguments

  float isoValue = -999;
  if( isoType_ == ISO_ECAL ){
    isoValue = elPat->ecalPFClusterIso();
  }else if( isoType_ == ISO_HCAL ){
    isoValue = elPat->hcalPFClusterIso();
  }else{
    throw cms::Exception("ERROR: unknown type requested for PF cluster isolation.")
      << std::endl << "Check VID configuration." << std::endl;
  }
  float isoValueCorr = std::max(0.0f, isoValue - rho*eA);
  
  // Apply the cut and return the result
  // Scale by pT if the relative isolation is requested but avoid division by 0
  return isoValueCorr < isoCut*(isRelativeIso_ ? cand->pt() : 1.);
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

  const float  eA = effectiveAreas_.getEffectiveArea( absEta );
  const float rho = rhoHandle_.isValid() ? (float)(*rhoHandle_) : 0; // std::max likes float arguments

  float isoValue = -999;
  if( isoType_ == ISO_ECAL ){
    isoValue = elPat->ecalPFClusterIso();
  }else if( isoType_ == ISO_HCAL ){
    isoValue = elPat->hcalPFClusterIso();
  }else{
    throw cms::Exception("ERROR: unknown type requested for PF cluster isolation.")
      << std::endl << "Check VID configuration." << std::endl;
  } 
  float isoValueCorr = std::max(0.0f, isoValue - rho*eA);
  
  // Divide by pT if the relative isolation is requested
  if( isRelativeIso_ )
    isoValueCorr /= ele->pt();

  // Apply the cut and return the result
  return isoValueCorr;
}
