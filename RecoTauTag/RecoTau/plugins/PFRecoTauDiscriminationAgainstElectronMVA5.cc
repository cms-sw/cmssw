/* class PFRecoTauDiscriminationAgainstElectronMVA5
 * created : Aug 13 2013,
 * revised : ,
 * Authorss : Ivo Naranjo (LLR Ecole-Polytechnique)
 */

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/interface/AntiElectronIDMVA5.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <TMath.h>

#include <iostream>
#include <sstream>
#include <fstream>

using namespace reco;

class PFRecoTauDiscriminationAgainstElectronMVA5 : public PFTauDiscriminationProducerBase  
{
 public:
  explicit PFRecoTauDiscriminationAgainstElectronMVA5(const edm::ParameterSet& cfg)
    : PFTauDiscriminationProducerBase(cfg),
      mva_(nullptr),
      category_output_()
  {
    mva_ = new AntiElectronIDMVA5(cfg);

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

  ~PFRecoTauDiscriminationAgainstElectronMVA5() override
  {
    delete mva_;
  }

private:
  bool isInEcalCrack(double) const;

  std::string moduleLabel_;

  AntiElectronIDMVA5* mva_;
  float* mvaInput_;

  edm::InputTag srcGsfElectrons_;
  edm::EDGetTokenT<reco::GsfElectronCollection> GsfElectrons_token;
  edm::Handle<reco::GsfElectronCollection> gsfElectrons_;
  edm::Handle<TauCollection> taus_;

  std::unique_ptr<PFTauDiscriminator> category_output_;

  int verbosity_;
};

void PFRecoTauDiscriminationAgainstElectronMVA5::beginEvent(const edm::Event& evt, const edm::EventSetup& es)
{
  mva_->beginEvent(evt, es);

  evt.getByToken(Tau_token, taus_);
  category_output_.reset(new PFTauDiscriminator(TauRefProd(taus_)));

  evt.getByToken(GsfElectrons_token, gsfElectrons_);
}

double PFRecoTauDiscriminationAgainstElectronMVA5::discriminate(const PFTauRef& thePFTauRef) const
{
  double mvaValue = 1.;
  double category = -1.;
  bool isGsfElectronMatched = false;

  float deltaRDummy = 9.9;

  float tauEtaAtEcalEntrance = -99.;
  float sumEtaTimesEnergy = 0.;
  float sumEnergy = 0.;
  const std::vector<reco::PFCandidatePtr>& signalPFCands = thePFTauRef->signalPFCands();
  for ( std::vector<reco::PFCandidatePtr>::const_iterator pfCandidate = signalPFCands.begin();
	pfCandidate != signalPFCands.end(); ++pfCandidate ) {
    sumEtaTimesEnergy += ((*pfCandidate)->positionAtECALEntrance().eta()*(*pfCandidate)->energy());
    sumEnergy += (*pfCandidate)->energy();
  }
  if ( sumEnergy > 0. ) {
    tauEtaAtEcalEntrance = sumEtaTimesEnergy/sumEnergy;
  }

  float leadChargedPFCandEtaAtEcalEntrance = -99.;
  float leadChargedPFCandPt = -99.;
  for ( std::vector<reco::PFCandidatePtr>::const_iterator pfCandidate = signalPFCands.begin();
	pfCandidate != signalPFCands.end(); ++pfCandidate ) {
    const reco::Track* track = nullptr;
    if ( (*pfCandidate)->trackRef().isNonnull() ) track = (*pfCandidate)->trackRef().get();
    else if ( (*pfCandidate)->muonRef().isNonnull() && (*pfCandidate)->muonRef()->innerTrack().isNonnull()  ) track = (*pfCandidate)->muonRef()->innerTrack().get();
    else if ( (*pfCandidate)->muonRef().isNonnull() && (*pfCandidate)->muonRef()->globalTrack().isNonnull() ) track = (*pfCandidate)->muonRef()->globalTrack().get();
    else if ( (*pfCandidate)->muonRef().isNonnull() && (*pfCandidate)->muonRef()->outerTrack().isNonnull()  ) track = (*pfCandidate)->muonRef()->outerTrack().get();
    else if ( (*pfCandidate)->gsfTrackRef().isNonnull() ) track = (*pfCandidate)->gsfTrackRef().get();
    if ( track ) {
      if ( track->pt() > leadChargedPFCandPt ) {
	leadChargedPFCandEtaAtEcalEntrance = (*pfCandidate)->positionAtECALEntrance().eta();
	leadChargedPFCandPt = track->pt();
      }
    }
  }

  if( (*thePFTauRef).leadPFChargedHadrCand().isNonnull()) {
    for ( reco::GsfElectronCollection::const_iterator theGsfElectron = gsfElectrons_->begin();
	  theGsfElectron != gsfElectrons_->end(); ++theGsfElectron ) {
      if ( theGsfElectron->pt() > 10. ) { // CV: only take electrons above some minimal energy/Pt into account...
	double deltaREleTau = deltaR(theGsfElectron->p4(), thePFTauRef->p4());
	deltaRDummy = deltaREleTau;
	if ( deltaREleTau < 0.3 ) {
	  double mva_match = mva_->MVAValue(*thePFTauRef, *theGsfElectron);
	  size_t numSignalPFGammaCands = thePFTauRef->signalPFGammaCands().size();
	  bool hasGsfTrack = thePFTauRef->leadPFChargedHadrCand()->gsfTrackRef().isNonnull();
  	    
	  //// Veto taus that go to Ecal crack
	  if ( isInEcalCrack(tauEtaAtEcalEntrance) || isInEcalCrack(leadChargedPFCandEtaAtEcalEntrance) ) {
	    // add category index
	    category_output_->setValue(tauIndex_, category);
	    // return MVA output value
	    return -99;
	  }
	  //// Veto taus that go to Ecal crack

	  if ( TMath::Abs(tauEtaAtEcalEntrance) < 1.479 ) { // Barrel
	    if        ( numSignalPFGammaCands == 0 && !hasGsfTrack ) {
	      category = 4.;
	    } else if ( numSignalPFGammaCands == 0 &&  hasGsfTrack ) {
	      category = 5.;
	    } else if ( numSignalPFGammaCands >= 1 && !hasGsfTrack ) {
	      category = 6.;
	    } else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack ) {
	      category = 7.;
	    }
	  } else { // Endcap
	    if        ( numSignalPFGammaCands == 0 && !hasGsfTrack ) {
	      category = 12.;
	    } else if ( numSignalPFGammaCands == 0 &&  hasGsfTrack ) {
	      category = 13.;
	    } else if ( numSignalPFGammaCands >= 1 && !hasGsfTrack ) {
	      category = 14.;
	    } else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack ) {
	      category = 15.;
	    }
	  }

	  mvaValue = TMath::Min(mvaValue, mva_match);
	  isGsfElectronMatched = true;
	} // deltaR < 0.3
      } // electron pt > 10
    } // end of loop over electrons

    if ( !isGsfElectronMatched ) {
      mvaValue = mva_->MVAValue(*thePFTauRef);
      size_t numSignalPFGammaCands = thePFTauRef->signalPFGammaCands().size();
      bool hasGsfTrack = thePFTauRef->leadPFChargedHadrCand()->gsfTrackRef().isNonnull();
      
      //// Veto taus that go to Ecal crack
      if ( isInEcalCrack(tauEtaAtEcalEntrance) || isInEcalCrack(leadChargedPFCandEtaAtEcalEntrance) ) {
	// add category index
	category_output_->setValue(tauIndex_, category);
	// return MVA output value
	return -99;
      }
      //// Veto taus that go to Ecal crack
      
      if ( TMath::Abs(tauEtaAtEcalEntrance) < 1.479 ) { // Barrel
	if        ( numSignalPFGammaCands == 0 && !hasGsfTrack ) {
	  category = 0.;
	} else if ( numSignalPFGammaCands == 0 &&  hasGsfTrack ) {
	  category = 1.;
	} else if ( numSignalPFGammaCands >= 1 && !hasGsfTrack ) {
	  category = 2.;
	} else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack ) {
	  category = 3.;
	}
      } else { // Endcap
	if        ( numSignalPFGammaCands == 0 && !hasGsfTrack ) {
	  category = 8.;
	} else if ( numSignalPFGammaCands == 0 &&  hasGsfTrack ) {
	  category = 9.;
	} else if ( numSignalPFGammaCands >= 1 && !hasGsfTrack ) {
	  category = 10.;
	} else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack ) {
	  category = 11.;
	}
      }
    }
  }

  if ( verbosity_ ) {
    edm::LogPrint("PFTauAgainstEleMVA5") << "<PFRecoTauDiscriminationAgainstElectronMVA5::discriminate>:" ;
    edm::LogPrint("PFTauAgainstEleMVA5") << " tau: Pt = " << thePFTauRef->pt() << ", eta = " << thePFTauRef->eta() << ", phi = " << thePFTauRef->phi();
    edm::LogPrint("PFTauAgainstEleMVA5") << " deltaREleTau = " << deltaRDummy << ", isGsfElectronMatched = " << isGsfElectronMatched;
    edm::LogPrint("PFTauAgainstEleMVA5") << " #Prongs = " << thePFTauRef->signalPFChargedHadrCands().size();
    edm::LogPrint("PFTauAgainstEleMVA5") << " MVA = " << mvaValue << ", category = " << category;
  }

  // add category index
  category_output_->setValue(tauIndex_, category);
  // return MVA output value
  return mvaValue;
}

void PFRecoTauDiscriminationAgainstElectronMVA5::endEvent(edm::Event& evt)
{
  // add all category indices to event
  evt.put(std::move(category_output_), "category");
}

bool
PFRecoTauDiscriminationAgainstElectronMVA5::isInEcalCrack(double eta) const
{
  double absEta = fabs(eta);
  return (absEta > 1.460 && absEta < 1.558);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstElectronMVA5);
