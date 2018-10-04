/* class PATTauDiscriminationAgainstElectronMVA6
 * created : Apr 14 2016,
 * revised : ,
 * Authorss :  Anne-Catherine Le Bihan (IPHC)
 */

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/interface/AntiElectronIDMVA6.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/PATTauDiscriminator.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <iostream>
#include <sstream>
#include <fstream>

using namespace pat;

class PATTauDiscriminationAgainstElectronMVA6 : public PATTauDiscriminationProducerBase  
{
 public:
  explicit PATTauDiscriminationAgainstElectronMVA6(const edm::ParameterSet& cfg)
    : PATTauDiscriminationProducerBase(cfg),
      mva_(),
      category_output_()
  {
    mva_ = std::make_unique<AntiElectronIDMVA6>(cfg);

    srcElectrons = cfg.getParameter<edm::InputTag>("srcElectrons");
    electronToken = consumes<pat::ElectronCollection>(srcElectrons);
    verbosity_ = ( cfg.exists("verbosity") ) ?
      cfg.getParameter<int>("verbosity") : 0;

    // add category index
    produces<PATTauDiscriminator>("category");
  }

  void beginEvent(const edm::Event&, const edm::EventSetup&) override;

  double discriminate(const TauRef&) const override;

  void endEvent(edm::Event&) override;

  ~PATTauDiscriminationAgainstElectronMVA6() override{}

private:
  bool isInEcalCrack(double) const;

  std::string moduleLabel_;
  std::unique_ptr<AntiElectronIDMVA6> mva_;
  
  edm::InputTag srcElectrons;
  edm::EDGetTokenT<pat::ElectronCollection> electronToken;
  edm::Handle<pat::ElectronCollection> Electrons;
  edm::Handle<TauCollection> taus_;

  std::unique_ptr<PATTauDiscriminator> category_output_;
		
  int verbosity_;
};

void PATTauDiscriminationAgainstElectronMVA6::beginEvent(const edm::Event& evt, const edm::EventSetup& es)
{
  mva_->beginEvent(evt, es);

  evt.getByToken(Tau_token, taus_);
  category_output_.reset(new PATTauDiscriminator(TauRefProd(taus_)));

  evt.getByToken(electronToken, Electrons);
}

double PATTauDiscriminationAgainstElectronMVA6::discriminate(const TauRef& theTauRef) const
{
  double mvaValue = 1.;
  double category = -1.;
  bool isGsfElectronMatched = false;
  float deltaRDummy = 9.9;
  const float ECALBarrelEndcapEtaBorder = 1.479;
  float tauEtaAtEcalEntrance = theTauRef->etaAtEcalEntrance();  
  float leadChargedPFCandEtaAtEcalEntrance = theTauRef->etaAtEcalEntranceLeadChargedCand();
      
  if( (*theTauRef).leadChargedHadrCand().isNonnull()) {
    int numSignalPFGammaCandsInSigCone = 0;
    const reco::CandidatePtrVector signalGammaCands = theTauRef->signalGammaCands();
    for ( const auto & gamma : signalGammaCands ) {
      double dR = deltaR(gamma->p4(), theTauRef->leadChargedHadrCand()->p4());
      double signalrad = std::max(0.05, std::min(0.10, 3.0/std::max(1.0, theTauRef->pt())));      
      // gammas inside the tau signal cone
      if (dR < signalrad) {
        numSignalPFGammaCandsInSigCone += 1;
      }
    }
    // loop over the electrons
    for ( const auto & theElectron : *Electrons ) {
      if ( theElectron.pt() > 10. ) { // CV: only take electrons above some minimal energy/Pt into account...	
	double deltaREleTau = deltaR(theElectron.p4(), theTauRef->p4());
	deltaRDummy = deltaREleTau;
	if( deltaREleTau < 0.3 ){ 	
	  double mva_match = mva_->MVAValue(*theTauRef, theElectron);
	  bool hasGsfTrack = false;
          pat::PackedCandidate const* packedLeadTauCand = dynamic_cast<pat::PackedCandidate const*>(theTauRef->leadChargedHadrCand().get());
          if( abs(packedLeadTauCand->pdgId()) == 11 ) 
	    hasGsfTrack = true;
	  if ( !hasGsfTrack )
            hasGsfTrack = theElectron.gsfTrack().isNonnull();

	  // veto taus that go to Ecal crack
	  if ( isInEcalCrack(tauEtaAtEcalEntrance) || isInEcalCrack(leadChargedPFCandEtaAtEcalEntrance) ) {
	    // add category index
	    category_output_->setValue(tauIndex_, category);
	    // return MVA output value
	    return -99;
	  }
	  // Veto taus that go to Ecal crack
	  if( std::abs(tauEtaAtEcalEntrance) < ECALBarrelEndcapEtaBorder ){ // Barrel
	    if( numSignalPFGammaCandsInSigCone == 0 && hasGsfTrack ){
	      category = 5.;
	    }
	    else if ( numSignalPFGammaCandsInSigCone >= 1 && hasGsfTrack ) {
	      category = 7.;
	    }
	  } else { // Endcap
	    if ( numSignalPFGammaCandsInSigCone == 0 && hasGsfTrack ) {
	      category = 13.;
	    }
	    else if ( numSignalPFGammaCandsInSigCone >= 1 && hasGsfTrack ) {
	      category = 15.;
	    }
	  }
	  mvaValue = std::min(mvaValue, mva_match);
	  isGsfElectronMatched = true;	
	 } // deltaR < 0.3
       } // electron pt > 10
     } // end of loop over electrons

    if ( !isGsfElectronMatched ) {
      mvaValue = mva_->MVAValue(*theTauRef);
      bool hasGsfTrack = false;
      pat::PackedCandidate const* packedLeadTauCand = dynamic_cast<pat::PackedCandidate const*>(theTauRef->leadChargedHadrCand().get());
      if( abs(packedLeadTauCand->pdgId()) == 11 ) hasGsfTrack = true;
          
      // veto taus that go to Ecal crack
      if ( isInEcalCrack(tauEtaAtEcalEntrance) || isInEcalCrack(leadChargedPFCandEtaAtEcalEntrance) ) {
	// add category index
	category_output_->setValue(tauIndex_, category);
	// return MVA output value
	return -99;
      }
      // veto taus that go to Ecal crack     
      if( std::abs(tauEtaAtEcalEntrance) < ECALBarrelEndcapEtaBorder ){ // Barrel
	if( numSignalPFGammaCandsInSigCone == 0 && !hasGsfTrack ){
	  category = 0.;
	}
	else if ( numSignalPFGammaCandsInSigCone >= 1 && !hasGsfTrack ) {
	  category = 2.;
	}
      } else { // Endcap
	if ( numSignalPFGammaCandsInSigCone == 0 && !hasGsfTrack ) {
	  category = 8.;
	}
	else if ( numSignalPFGammaCandsInSigCone >= 1 && !hasGsfTrack ) {
	  category = 10.;
	}
      }
    }
  }
  if ( verbosity_ ) {
    edm::LogPrint("PATTauAgainstEleMVA6") << "<PATTauDiscriminationAgainstElectronMVA6::discriminate>:" ;
    edm::LogPrint("PATTauAgainstEleMVA6") << " tau: Pt = " << theTauRef->pt() << ", eta = " << theTauRef->eta() << ", phi = " << theTauRef->phi();
    edm::LogPrint("PATTauAgainstEleMVA6") << " deltaREleTau = " << deltaRDummy << ", isGsfElectronMatched = " << isGsfElectronMatched;
    edm::LogPrint("PATTauAgainstEleMVA6") << " #Prongs = " << theTauRef->signalChargedHadrCands().size();
    edm::LogPrint("PATTauAgainstEleMVA6") << " MVA = " << mvaValue << ", category = " << category;
  }
  // add category index
  category_output_->setValue(tauIndex_, category);
  // return MVA output value
  return mvaValue;
}

void PATTauDiscriminationAgainstElectronMVA6::endEvent(edm::Event& evt)
{
  // add all category indices to event
  evt.put(std::move(category_output_), "category");
}

bool
PATTauDiscriminationAgainstElectronMVA6::isInEcalCrack(double eta) const
{
  double absEta = fabs(eta);
  return (absEta > 1.460 && absEta < 1.558);
}

DEFINE_FWK_MODULE(PATTauDiscriminationAgainstElectronMVA6);
