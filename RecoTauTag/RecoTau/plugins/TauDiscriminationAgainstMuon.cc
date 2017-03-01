
/* 
 * class TauDiscriminationAgainstMuon
 * created : July 09 2010
 * revised : 
 * Authors : Sho Maruyama, Christian Veelken (UC Davis)
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <string>

namespace {
using namespace reco;

template<class TauType, class TauDiscriminator>
class TauDiscriminationAgainstMuon final : public TauDiscriminationProducerBase<TauType, TauDiscriminator>
{
 public:
  // setup framework types for this tautype
  typedef std::vector<TauType>    TauCollection; 
  typedef edm::Ref<TauCollection> TauRef;    

  explicit TauDiscriminationAgainstMuon(const edm::ParameterSet&);
  ~TauDiscriminationAgainstMuon() {} 

  // called at the beginning of every event
  void beginEvent(const edm::Event&, const edm::EventSetup&) override;

  double discriminate(const TauRef&) const override;

 private:  
  bool evaluateMuonVeto(const reco::Muon&) const;

  edm::InputTag muonSource_;
  edm::Handle<reco::MuonCollection> muons_;
  double dRmatch_;

  enum { kNoSegMatch, kTwoDCut, kMerePresence, kCombined };
  int discriminatorOption_;

  double coeffCaloComp_;
  double coeffSegmComp_;
  double muonCompCut_;
};

template<class TauType, class TauDiscriminator>
TauDiscriminationAgainstMuon<TauType, TauDiscriminator>::TauDiscriminationAgainstMuon(const edm::ParameterSet& cfg)
  : TauDiscriminationProducerBase<TauType, TauDiscriminator>(cfg) 
{
  //if ( cfg.exists("muonSource") ) muonSource_ = cfg.getParameter<edm::InputTag>("muonSource");
  muonSource_ = cfg.getParameter<edm::InputTag>("muonSource");
  dRmatch_ = ( cfg.exists("dRmatch") ) ? cfg.getParameter<double>("dRmatch") : 0.5;

  std::string discriminatorOption_string = cfg.getParameter<std::string>("discriminatorOption");  
  if      ( discriminatorOption_string == "noSegMatch"   ) discriminatorOption_ = kNoSegMatch;
  else if ( discriminatorOption_string == "twoDCut"      ) discriminatorOption_ = kTwoDCut;
  else if ( discriminatorOption_string == "merePresence" ) discriminatorOption_ = kMerePresence;
  else if ( discriminatorOption_string == "combined"     ) discriminatorOption_ = kCombined;
  else {
    throw edm::Exception(edm::errors::UnimplementedFeature) << " Invalid Discriminator Option! Please check cfi file \n";
  }

  coeffCaloComp_ = cfg.getParameter<double>("caloCompCoefficient");
  coeffSegmComp_ = cfg.getParameter<double>("segmCompCoefficient");
  muonCompCut_ = cfg.getParameter<double>("muonCompCut");
}

template<class TauType, class TauDiscriminator>
void TauDiscriminationAgainstMuon<TauType, TauDiscriminator>::beginEvent(const edm::Event& evt, const edm::EventSetup& evtSetup)
{
  evt.getByLabel(muonSource_, muons_);	
}		

template<class TauType, class TauDiscriminator>
bool TauDiscriminationAgainstMuon<TauType, TauDiscriminator>::evaluateMuonVeto(const reco::Muon& muon) const 
{
  bool decision = true;	

  if ( discriminatorOption_ == kNoSegMatch ) {
    if ( muon.numberOfMatches() > 0 ) decision = false;
  } else if ( discriminatorOption_ == kTwoDCut ) {
    double segmComp = muon::segmentCompatibility(muon);
    double caloComp = muon.caloCompatibility(); 
    if ( (coeffCaloComp_*segmComp + coeffSegmComp_*caloComp) > muonCompCut_ ) decision = false;
  } else if ( discriminatorOption_ == kMerePresence ) {
    decision = false;
  } else if ( discriminatorOption_ == kCombined ) { // testing purpose only
    unsigned int muonType = 0;
    if      ( muon.isGlobalMuon()  ) muonType = 1;
    else if ( muon.isCaloMuon()    ) muonType = 2;
    else if ( muon.isTrackerMuon() ) muonType = 3;

    bool eta_veto = ( fabs(muon.eta()) > 2.3 || (fabs(muon.eta()) > 1.4 && fabs(muon.eta()) < 1.6) ) ? true : false;
    bool phi_veto = ( muon.phi() < 0.1 && muon.phi() > -0.1 ) ? true : false;

    if ( muonType != 1 || muon.numberOfMatches() > 0 || eta_veto || phi_veto ) decision = false;
  }

  return decision;
}

template<class TauType, class TauDiscriminator>
double TauDiscriminationAgainstMuon<TauType, TauDiscriminator>::discriminate(const TauRef& tau) const 
{
  bool decision = true;	

  for ( reco::MuonCollection::const_iterator muon = muons_->begin();
	muon != muons_->end(); ++muon ) {
    if ( reco::deltaR(muon->p4(), tau->p4()) < dRmatch_ ) decision &= evaluateMuonVeto(*muon);
  }

  return (decision ? 1. : 0.);
}
}

#include "FWCore/Framework/interface/MakerMacros.h"

//typedef TauDiscriminationAgainstMuon<PFTau, PFTauDiscriminator> PFRecoTauDiscriminationAgainstMuon;
typedef TauDiscriminationAgainstMuon<CaloTau, CaloTauDiscriminator> CaloRecoTauDiscriminationAgainstMuon;

//DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstMuon);
DEFINE_FWK_MODULE(CaloRecoTauDiscriminationAgainstMuon);
