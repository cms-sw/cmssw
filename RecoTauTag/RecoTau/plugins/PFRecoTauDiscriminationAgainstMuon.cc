/* 
 * class PFRecoTauDiscriminationAgainstMuon
 * created : May 07 2008,
 * revised : always,
 * Authors : Sho Maruyama,M.Bachtis
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "DataFormats/MuonReco/interface/MuonSelectors.h"


#include <string>

using namespace reco;

class PFRecoTauDiscriminationAgainstMuon : public PFTauDiscriminationProducerBase 
{
 public:
  explicit PFRecoTauDiscriminationAgainstMuon(const edm::ParameterSet& iConfig)
    : PFTauDiscriminationProducerBase(iConfig) 
  {   
    discriminatorOption_  = iConfig.getParameter<std::string>("discriminatorOption");  
    hop_  = iConfig.getParameter<double>("HoPMin");  
    a  = iConfig.getParameter<double>("a");  
    b  = iConfig.getParameter<double>("b");  
    c  = iConfig.getParameter<double>("c");  	 
  }

  ~PFRecoTauDiscriminationAgainstMuon() {} 

  double discriminate(const PFTauRef& pfTau);

 private:  
  std::string discriminatorOption_;
  double hop_;
  double a;
  double b;
  double c;
};

double PFRecoTauDiscriminationAgainstMuon::discriminate(const PFTauRef& thePFTauRef)
{
  bool decision = true;

  bool applyCaloMuonCut = false;

  if ( thePFTauRef->hasMuonReference() ) {

    MuonRef muonref = thePFTauRef->leadPFChargedHadrCand()->muonRef();
    if ( discriminatorOption_ == "noSegMatch" ) {
      if ( muonref ->numberOfMatches() > 0 ) decision = false;
    } else if (discriminatorOption_ == "twoDCut") {
      double seg = muon::segmentCompatibility(*muonref);
      double calo= muonref->caloCompatibility(); 
      double border = calo * a + seg * b +c;
      if ( border > 0 ) decision = false; 
    } else if ( discriminatorOption_ == "merePresence" ) {
      decision = false;
    } else if (discriminatorOption_ == "combined" ) { // testing purpose only
      unsigned int muType = 0;
      if      ( muonref->isGlobalMuon()  ) muType = 1;
      else if ( muonref->isCaloMuon()    ) muType = 2;
      else if ( muonref->isTrackerMuon() ) muType = 3;
      double muonEnergyFraction = thePFTauRef->pfTauTagInfoRef()->pfjetRef()->chargedMuEnergyFraction();
      bool eta_veto = false;
      bool phi_veto = false;
      if ( fabs(muonref->eta()) > 2.3 || (fabs(muonref->eta()) > 1.4 && fabs(muonref->eta()) < 1.6)) eta_veto = true;
      if ( muonref->phi() < 0.1 && muonref->phi() > -0.1) phi_veto = true;
      if ( muType != 1 || muonref ->numberOfMatches() > 0 || eta_veto || phi_veto || muonEnergyFraction > 0.9 ) decision = false; // as place holder
    } else if ( discriminatorOption_ == "noAllArbitrated" || discriminatorOption_ == "noAllArbitratedWithHOP" ) { 
      if ( muon::isGoodMuon(*muonref, muon::AllArbitrated) ) decision = false;
      if ( discriminatorOption_ == "noAllArbitratedWithHOP" ) applyCaloMuonCut = true;
    } else if ( discriminatorOption_ == "HOP" ) { 
      applyCaloMuonCut = true;
    } else {
      throw edm::Exception(edm::errors::UnimplementedFeature) 
	<< " Invalid Discriminator option = " << discriminatorOption_ << " --> please check cfi file !!\n";
    }
  } // valid muon ref

  // Additional calo. muon cut: veto one prongs compatible with MIP signature
  if ( applyCaloMuonCut ) {
    if ( thePFTauRef->leadPFChargedHadrCand().isNonnull() ) {
      double muonCaloEn = thePFTauRef->leadPFChargedHadrCand()->hcalEnergy() + thePFTauRef->leadPFChargedHadrCand()->ecalEnergy();
      if ( thePFTauRef->decayMode() == 0 && (muonCaloEn/thePFTauRef->leadPFChargedHadrCand()->p()) < hop_ ) decision = false;
    }
  }
  
  return (decision ? 1. : 0.);
} 

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstMuon);
