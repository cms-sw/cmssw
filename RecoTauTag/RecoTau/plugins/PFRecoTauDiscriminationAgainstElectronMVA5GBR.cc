/* class PFRecoTauDiscriminationAgainstElectronMVA5
 * created : Aug 13 2013,
 * revised : ,
 * Authorss : Ivo Naranjo (LLR Ecole-Polytechnique)
 */

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/interface/AntiElectronIDMVA5GBR.h"

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

class PFRecoTauDiscriminationAgainstElectronMVA5GBR : public PFTauDiscriminationProducerBase  {
public:
  explicit PFRecoTauDiscriminationAgainstElectronMVA5GBR(const edm::ParameterSet& iConfig)
    : PFTauDiscriminationProducerBase(iConfig),
      mva_(0),
      category_output_(0)
  {
    method_  = iConfig.getParameter<std::string>("method");
    gbrFile_ = iConfig.getParameter<edm::FileInPath>("gbrFile");
    if ( gbrFile_.location() == edm::FileInPath::Unknown) 
      throw cms::Exception("PFRecoTauDiscriminationAgainstElectronMVA5GBR") 
	<< " Failed to find File = " << gbrFile_ << " !!\n";

    returnMVA_                  = iConfig.getParameter<bool>("returnMVA");
    minMVANoEleMatchWOgWOgsfBL_ = iConfig.getParameter<double>("minMVANoEleMatchWOgWOgsfBL");
    minMVANoEleMatchWOgWgsfBL_  = iConfig.getParameter<double>("minMVANoEleMatchWOgWgsfBL");
    minMVANoEleMatchWgWOgsfBL_  = iConfig.getParameter<double>("minMVANoEleMatchWgWOgsfBL");
    minMVANoEleMatchWgWgsfBL_   = iConfig.getParameter<double>("minMVANoEleMatchWgWgsfBL");
    minMVAWOgWOgsfBL_           = iConfig.getParameter<double>("minMVAWOgWOgsfBL");
    minMVAWOgWgsfBL_            = iConfig.getParameter<double>("minMVAWOgWgsfBL");
    minMVAWgWOgsfBL_            = iConfig.getParameter<double>("minMVAWgWOgsfBL");
    minMVAWgWgsfBL_             = iConfig.getParameter<double>("minMVAWgWgsfBL");
    minMVANoEleMatchWOgWOgsfEC_ = iConfig.getParameter<double>("minMVANoEleMatchWOgWOgsfEC");
    minMVANoEleMatchWOgWgsfEC_  = iConfig.getParameter<double>("minMVANoEleMatchWOgWgsfEC");
    minMVANoEleMatchWgWOgsfEC_  = iConfig.getParameter<double>("minMVANoEleMatchWgWOgsfEC");
    minMVANoEleMatchWgWgsfEC_   = iConfig.getParameter<double>("minMVANoEleMatchWgWgsfEC");
    minMVAWOgWOgsfEC_           = iConfig.getParameter<double>("minMVAWOgWOgsfEC");
    minMVAWOgWgsfEC_            = iConfig.getParameter<double>("minMVAWOgWgsfEC");
    minMVAWgWOgsfEC_            = iConfig.getParameter<double>("minMVAWgWOgsfEC");
    minMVAWgWgsfEC_             = iConfig.getParameter<double>("minMVAWgWgsfEC");

    srcGsfElectrons_ = iConfig.getParameter<edm::InputTag>("srcGsfElectrons");
    GsfElectrons_token = consumes<reco::GsfElectronCollection>(srcGsfElectrons_);

    mva_ = new AntiElectronIDMVA5GBR();
    mva_->Initialize_from_file(method_, gbrFile_.fullPath().data());

    verbosity_ = ( iConfig.exists("verbosity") ) ?
      iConfig.getParameter<int>("verbosity") : 0;

    // add category index
    if ( returnMVA_ ) {
      produces<PFTauDiscriminator>("category");
    }
  }

  void beginEvent(const edm::Event&, const edm::EventSetup&);

  double discriminate(const PFTauRef&);

  void endEvent(edm::Event&);

  ~PFRecoTauDiscriminationAgainstElectronMVA5GBR()
  {
    delete mva_;
  }

private:
  bool isInEcalCrack(double) const;

  std::string method_ ;
  edm::FileInPath gbrFile_;
  AntiElectronIDMVA5GBR* mva_;
  bool returnMVA_ ;
  double minMVANoEleMatchWOgWOgsfBL_ ;
  double minMVANoEleMatchWOgWgsfBL_ ;
  double minMVANoEleMatchWgWOgsfBL_ ;
  double minMVANoEleMatchWgWgsfBL_ ;
  double minMVAWOgWOgsfBL_ ;
  double minMVAWOgWgsfBL_ ;
  double minMVAWgWOgsfBL_ ;
  double minMVAWgWgsfBL_ ;
  double minMVANoEleMatchWOgWOgsfEC_ ;
  double minMVANoEleMatchWOgWgsfEC_ ;
  double minMVANoEleMatchWgWOgsfEC_ ;
  double minMVANoEleMatchWgWgsfEC_ ;
  double minMVAWOgWOgsfEC_ ;
  double minMVAWOgWgsfEC_ ;
  double minMVAWgWOgsfEC_ ;
  double minMVAWgWgsfEC_ ;
  edm::InputTag srcGsfElectrons_;
  edm::EDGetTokenT<reco::GsfElectronCollection> GsfElectrons_token;
  edm::Handle<reco::GsfElectronCollection> gsfElectrons_;
  edm::Handle<TauCollection> taus_;
  std::auto_ptr<PFTauDiscriminator> category_output_;
  size_t tauIndex_;

  int verbosity_;
};

void PFRecoTauDiscriminationAgainstElectronMVA5GBR::beginEvent(const edm::Event& evt, const edm::EventSetup& es)
{
  if ( returnMVA_ ) {
    evt.getByToken(Tau_token, taus_);
    category_output_.reset(new PFTauDiscriminator(TauRefProd(taus_)));
    tauIndex_ = 0;
  }
  evt.getByToken(GsfElectrons_token, gsfElectrons_);
}

double PFRecoTauDiscriminationAgainstElectronMVA5GBR::discriminate(const PFTauRef& thePFTauRef)
{
  double mva = 1.;
  double workingPoint = 1.;
  double category = -1.;
  bool isGsfElectronMatched = false;

  float deltaRDummy = 9.9;
  float mvaCutDummy = 999;

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
    const reco::Track* track = 0;
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
	  double workingPoint_match = 0.;
	  size_t numSignalPFGammaCands = thePFTauRef->signalPFGammaCands().size();
	  bool hasGsfTrack = thePFTauRef->leadPFChargedHadrCand()->gsfTrackRef().isNonnull();
  	    
	  //// Veto taus that go to Ecal crack
	  if ( isInEcalCrack(tauEtaAtEcalEntrance) || isInEcalCrack(leadChargedPFCandEtaAtEcalEntrance) ) {
	    if ( returnMVA_ ) {
	      // add category index
	      category_output_->setValue(tauIndex_, category);
	      ++tauIndex_;
	      // return MVA output value
	      return -99;
	    } else {
	      //return Workingpoint 0
	      return 0;
	    }
	  }
	  //// Veto taus that go to Ecal crack

	  double mvaCut = 999.;
	  if ( TMath::Abs(tauEtaAtEcalEntrance) < 1.479 ) { // Barrel
	    if        ( numSignalPFGammaCands == 0 && !hasGsfTrack ) {
	      category = 4.;
	      mvaCut = minMVAWOgWOgsfBL_;
	    } else if ( numSignalPFGammaCands == 0 &&  hasGsfTrack ) {
	      category = 5.;
	      mvaCut = minMVAWOgWgsfBL_;
	    } else if ( numSignalPFGammaCands >= 1 && !hasGsfTrack ) {
	      category = 6.;
	      mvaCut = minMVAWgWOgsfBL_;
	    } else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack ) {
	      category = 7.;
	      mvaCut = minMVAWgWgsfBL_;
	    }
	  } else { // Endcap
	    if        ( numSignalPFGammaCands == 0 && !hasGsfTrack ) {
	      category = 12.;
	      mvaCut = minMVAWOgWOgsfEC_;
	    } else if ( numSignalPFGammaCands == 0 &&  hasGsfTrack ) {
	      category = 13.;
	      mvaCut = minMVAWOgWgsfEC_;
	    } else if ( numSignalPFGammaCands >= 1 && !hasGsfTrack ) {
	      category = 14.;
	      mvaCut = minMVAWgWOgsfEC_;
	    } else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack ) {
	      category = 15.;
	      mvaCut = minMVAWgWgsfEC_;
	    }
	  }
	  workingPoint_match = (mva_match > mvaCut);
	  mvaCutDummy = mvaCut;

	  mva = TMath::Min(mva, mva_match);
	  workingPoint = TMath::Min(workingPoint, workingPoint_match);
	  isGsfElectronMatched = true;
	} // deltaR < 0.3
      } // electron pt > 10
    } // end of loop over electrons

    if ( !isGsfElectronMatched ) {
      mva = mva_->MVAValue(*thePFTauRef);
      size_t numSignalPFGammaCands = thePFTauRef->signalPFGammaCands().size();
      bool hasGsfTrack = thePFTauRef->leadPFChargedHadrCand()->gsfTrackRef().isNonnull();
      
      //// Veto taus that go to Ecal crack
      if ( isInEcalCrack(tauEtaAtEcalEntrance) || isInEcalCrack(leadChargedPFCandEtaAtEcalEntrance) ) {
	if ( returnMVA_ ) {
	  // add category index
	  category_output_->setValue(tauIndex_, category);
	  ++tauIndex_;
	  // return MVA output value
	  return -99;
	} else {
	  //return Workingpoint 0
	  return 0;
	}
      }
      //// Veto taus that go to Ecal crack
      
      double mvaCut = 999.;
      if ( TMath::Abs(tauEtaAtEcalEntrance) < 1.479 ) { // Barrel
	if        ( numSignalPFGammaCands == 0 && !hasGsfTrack ) {
	  category = 0.;
	  mvaCut = minMVANoEleMatchWOgWOgsfBL_;
	} else if ( numSignalPFGammaCands == 0 &&  hasGsfTrack ) {
	  category = 1.;
	  mvaCut = minMVANoEleMatchWOgWgsfBL_;
	} else if ( numSignalPFGammaCands >= 1 && !hasGsfTrack ) {
	  category = 2.;
	  mvaCut = minMVANoEleMatchWgWOgsfBL_;
	} else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack ) {
	  category = 3.;
	  mvaCut = minMVANoEleMatchWgWgsfBL_;
	}
      } else { // Endcap
	if        ( numSignalPFGammaCands == 0 && !hasGsfTrack ) {
	  category = 8.;
	  mvaCut = minMVANoEleMatchWOgWOgsfEC_;
	} else if ( numSignalPFGammaCands == 0 &&  hasGsfTrack ) {
	  category = 9.;
	  mvaCut = minMVANoEleMatchWOgWgsfEC_;
	} else if ( numSignalPFGammaCands >= 1 && !hasGsfTrack ) {
	  category = 10.;
	  mvaCut = minMVANoEleMatchWgWOgsfEC_;
	} else if ( numSignalPFGammaCands >= 1 &&  hasGsfTrack ) {
	  category = 11.;
	  mvaCut = minMVANoEleMatchWgWgsfEC_;
	}
      }
      workingPoint = (mva > mvaCut);
      mvaCutDummy = mvaCut;
    }
  }

  if ( verbosity_ ) {
    edm::LogPrint("PFTauAgainstEleMVA5") <<" Taus : "<<TauProducer_;
    edm::LogPrint("PFTauAgainstEleMVA5") << "<PFRecoTauDiscriminationAgainstElectronMVA5::discriminate>:" ;
    edm::LogPrint("PFTauAgainstEleMVA5") << " tau: Pt = " << thePFTauRef->pt() << ", eta = " << thePFTauRef->eta() << ", phi = " << thePFTauRef->phi() ;
    edm::LogPrint("PFTauAgainstEleMVA5") << " mva = " << mva << ", mvaCut = " << mvaCutDummy << ", isGsfElectronMatched = " << isGsfElectronMatched ;
    edm::LogPrint("PFTauAgainstEleMVA5") << " category = " << category << ": workingPoint = " << workingPoint ;
    edm::LogPrint("PFTauAgainstEleMVA5") << " deltaREleTau = " << deltaRDummy ;
    edm::LogPrint("PFTauAgainstEleMVA5") << " charged hadron in tau: "<<(*thePFTauRef).leadPFChargedHadrCand().isNonnull() ;
    edm::LogPrint("PFTauAgainstEleMVA5") << " Prongs in tau: " << thePFTauRef->signalPFChargedHadrCands().size() ;
    edm::LogPrint("PFTauAgainstEleMVA5") << " MVA GBR:" << mva ;
  }

  if ( returnMVA_ ) {
    // add category index
    category_output_->setValue(tauIndex_, category);
    ++tauIndex_;
    // return MVA output value
    return mva;
  } else {
    return workingPoint;
  }
}

void PFRecoTauDiscriminationAgainstElectronMVA5GBR::endEvent(edm::Event& evt)
{
  // add all category indices to event
  if ( returnMVA_ ) {
    evt.put(category_output_, "category");
  }
}

bool
PFRecoTauDiscriminationAgainstElectronMVA5GBR::isInEcalCrack(double eta) const
{
  double absEta = fabs(eta);
  return (absEta > 1.460 && absEta < 1.558);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstElectronMVA5GBR);
