/* class PFRecoTauDiscriminationAgainstElectronMVA6
 * created : Nov 2 2015,
 * revised : ,
 * Authorss : Fabio Colombo (KIT)
 */

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/interface/AntiElectronIDMVA6.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <iostream>
#include <sstream>
#include <fstream>

using namespace reco;

class PFRecoTauDiscriminationAgainstElectronMVA6 : public PFTauDiscriminationProducerBase  
{
 public:
  explicit PFRecoTauDiscriminationAgainstElectronMVA6(const edm::ParameterSet& cfg)
    : PFTauDiscriminationProducerBase(cfg),
      mva_(),
      category_output_()
  {
    mva_ = std::make_unique<AntiElectronIDMVA6>(cfg);

    srcGsfElectrons_ = cfg.getParameter<edm::InputTag>("srcGsfElectrons");
    GsfElectrons_token = consumes<reco::GsfElectronCollection>(srcGsfElectrons_);

    verbosity_ = ( cfg.exists("verbosity") ) ?
      cfg.getParameter<int>("verbosity") : 0;

    // add category index
    produces<PFTauDiscriminator>("category");
  }

  void beginEvent(const edm::Event&, const edm::EventSetup&) override;

  double discriminate(const PFTauRef&) const override;

  void endEvent(edm::Event&) override;

  ~PFRecoTauDiscriminationAgainstElectronMVA6() override {}

private:
  bool isInEcalCrack(double) const;

  std::string moduleLabel_;
  std::unique_ptr<AntiElectronIDMVA6> mva_;

  edm::InputTag srcGsfElectrons_;
  edm::EDGetTokenT<reco::GsfElectronCollection> GsfElectrons_token;
  edm::Handle<reco::GsfElectronCollection> gsfElectrons_;
  edm::Handle<TauCollection> taus_;

  std::unique_ptr<PFTauDiscriminator> category_output_;

  int verbosity_;
};

void PFRecoTauDiscriminationAgainstElectronMVA6::beginEvent(const edm::Event& evt, const edm::EventSetup& es)
{
  mva_->beginEvent(evt, es);

  evt.getByToken(Tau_token, taus_);
  category_output_.reset(new PFTauDiscriminator(TauRefProd(taus_)));

  evt.getByToken(GsfElectrons_token, gsfElectrons_);
}

double PFRecoTauDiscriminationAgainstElectronMVA6::discriminate(const PFTauRef& thePFTauRef) const
{
  double mvaValue = 1.;
  double category = -1.;
  bool isGsfElectronMatched = false;

  float deltaRDummy = 9.9;

  const float ECALBarrelEndcapEtaBorder = 1.479;
  float tauEtaAtEcalEntrance = -99.;
  float sumEtaTimesEnergy = 0.;
  float sumEnergy = 0.;
  const std::vector<reco::PFCandidatePtr>& signalPFCands = thePFTauRef->signalPFCands();
  for ( const auto & pfCandidate : signalPFCands ) {
    sumEtaTimesEnergy += (pfCandidate->positionAtECALEntrance().eta()*pfCandidate->energy());
    sumEnergy += pfCandidate->energy();
  }
  if ( sumEnergy > 0. ) {
    tauEtaAtEcalEntrance = sumEtaTimesEnergy/sumEnergy;
  }

  float leadChargedPFCandEtaAtEcalEntrance = -99.;
  float leadChargedPFCandPt = -99.;
  for ( const auto & pfCandidate : signalPFCands ) {
    const reco::Track* track = nullptr;
    if ( pfCandidate->trackRef().isNonnull() ) track = pfCandidate->trackRef().get();
    else if ( pfCandidate->muonRef().isNonnull() && pfCandidate->muonRef()->innerTrack().isNonnull()  ) track = pfCandidate->muonRef()->innerTrack().get();
    else if ( pfCandidate->muonRef().isNonnull() && pfCandidate->muonRef()->globalTrack().isNonnull() ) track = pfCandidate->muonRef()->globalTrack().get();
    else if ( pfCandidate->muonRef().isNonnull() && pfCandidate->muonRef()->outerTrack().isNonnull()  ) track = pfCandidate->muonRef()->outerTrack().get();
    else if ( pfCandidate->gsfTrackRef().isNonnull() ) track = pfCandidate->gsfTrackRef().get();
    if ( track ) {
      if ( track->pt() > leadChargedPFCandPt ) {
	leadChargedPFCandEtaAtEcalEntrance = pfCandidate->positionAtECALEntrance().eta();
	leadChargedPFCandPt = track->pt();
      }
    }
  }

  if( (*thePFTauRef).leadPFChargedHadrCand().isNonnull()) {

    int numSignalPFGammaCandsInSigCone = 0;
    const std::vector<reco::PFCandidatePtr>& signalPFGammaCands = thePFTauRef->signalPFGammaCands();
    
    for ( const auto & pfGamma : signalPFGammaCands ) {
            
      double dR = deltaR(pfGamma->p4(), thePFTauRef->leadPFChargedHadrCand()->p4());
      double signalrad = std::max(0.05, std::min(0.10, 3.0/std::max(1.0, thePFTauRef->pt())));
            
      // pfGammas inside the tau signal cone
      if (dR < signalrad) {
        numSignalPFGammaCandsInSigCone += 1;
      }
    }
    
    // loop over the electrons
    for ( const auto & theGsfElectron : *gsfElectrons_ ) {
      if ( theGsfElectron.pt() > 10. ) { // CV: only take electrons above some minimal energy/Pt into account...
	double deltaREleTau = deltaR(theGsfElectron.p4(), thePFTauRef->p4());
	deltaRDummy = deltaREleTau;
	if ( deltaREleTau < 0.3 ) {
	  double mva_match = mva_->MVAValue(*thePFTauRef, theGsfElectron);
	  bool hasGsfTrack = thePFTauRef->leadPFChargedHadrCand()->gsfTrackRef().isNonnull();
	  if ( !hasGsfTrack )
            hasGsfTrack = theGsfElectron.gsfTrack().isNonnull();

	  //// Veto taus that go to Ecal crack
	  if ( isInEcalCrack(tauEtaAtEcalEntrance) || isInEcalCrack(leadChargedPFCandEtaAtEcalEntrance) ) {
	    // add category index
	    category_output_->setValue(tauIndex_, category);
	    // return MVA output value
	    return -99;
	  }
	  //// Veto taus that go to Ecal crack

	  if ( std::abs(tauEtaAtEcalEntrance) < ECALBarrelEndcapEtaBorder ) { // Barrel
	    if ( numSignalPFGammaCandsInSigCone == 0 && hasGsfTrack ) {
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
      mvaValue = mva_->MVAValue(*thePFTauRef);
      bool hasGsfTrack = thePFTauRef->leadPFChargedHadrCand()->gsfTrackRef().isNonnull();
      
      //// Veto taus that go to Ecal crack
      if ( isInEcalCrack(tauEtaAtEcalEntrance) || isInEcalCrack(leadChargedPFCandEtaAtEcalEntrance) ) {
	// add category index
	category_output_->setValue(tauIndex_, category);
	// return MVA output value
	return -99;
      }
      //// Veto taus that go to Ecal crack
      
      if ( std::abs(tauEtaAtEcalEntrance) < ECALBarrelEndcapEtaBorder ) { // Barrel
	if ( numSignalPFGammaCandsInSigCone == 0 && !hasGsfTrack ) {
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
    edm::LogPrint("PFTauAgainstEleMVA6") << "<PFRecoTauDiscriminationAgainstElectronMVA6::discriminate>:" ;
    edm::LogPrint("PFTauAgainstEleMVA6") << " tau: Pt = " << thePFTauRef->pt() << ", eta = " << thePFTauRef->eta() << ", phi = " << thePFTauRef->phi();
    edm::LogPrint("PFTauAgainstEleMVA6") << " deltaREleTau = " << deltaRDummy << ", isGsfElectronMatched = " << isGsfElectronMatched;
    edm::LogPrint("PFTauAgainstEleMVA6") << " #Prongs = " << thePFTauRef->signalPFChargedHadrCands().size();
    edm::LogPrint("PFTauAgainstEleMVA6") << " MVA = " << mvaValue << ", category = " << category;
  }

  // add category index
  category_output_->setValue(tauIndex_, category);
  // return MVA output value
  return mvaValue;
}

void PFRecoTauDiscriminationAgainstElectronMVA6::endEvent(edm::Event& evt)
{
  // add all category indices to event
  evt.put(std::move(category_output_), "category");
}

bool
PFRecoTauDiscriminationAgainstElectronMVA6::isInEcalCrack(double eta) const
{
  double absEta = fabs(eta);
  return (absEta > 1.460 && absEta < 1.558);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstElectronMVA6);
