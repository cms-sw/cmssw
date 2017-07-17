#include "PFEGammaCandidateChecker.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidatePhotonExtra.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;
using namespace edm;
using namespace reco;

// we always assume we a looping over a 
namespace {
  struct gsfequals {
    reco::GsfTrackRef mytrack;
    gsfequals(const reco::PFCandidate& tofind) : mytrack(tofind.gsfTrackRef())
    {}
    bool operator()( const reco::PFCandidate& c ) const {
      if( c.gsfTrackRef().isNonnull() ) {
	return ( c.gsfTrackRef()->ptMode() == mytrack->ptMode() &&
		 c.gsfTrackRef()->etaMode() == mytrack->etaMode() );
      }
      return false;
    }
  };
  struct scequals {
    reco::SuperClusterRef mysc;
    scequals(const reco::PFCandidate& tofind) {
      if( tofind.photonRef().isNonnull() ) {
	mysc = tofind.photonRef()->superCluster();
      }
    }
    bool operator()( const reco::PFCandidate& c ) const {
      return ( mysc.isNonnull() && c.egammaExtraRef().isNonnull() && 
	       ( c.egammaExtraRef()->superClusterPFECALRef()->seed() == 
		 mysc->seed() ) );
    }
  };
}


PFEGammaCandidateChecker::PFEGammaCandidateChecker(const edm::ParameterSet& iConfig) {
  


  inputTagPFCandidatesReco_ 
    = iConfig.getParameter<InputTag>("pfCandidatesReco");

  inputTagPFCandidatesReReco_ 
    = iConfig.getParameter<InputTag>("pfCandidatesReReco");

  inputTagPFJetsReco_ 
    = iConfig.getParameter<InputTag>("pfJetsReco");

  inputTagPFJetsReReco_ 
    = iConfig.getParameter<InputTag>("pfJetsReReco");

  deltaEMax_ 
    = iConfig.getParameter<double>("deltaEMax");

  deltaEtaMax_ 
    = iConfig.getParameter<double>("deltaEtaMax");

  deltaPhiMax_ 
    = iConfig.getParameter<double>("deltaPhiMax");

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);

  printBlocks_ = 
    iConfig.getUntrackedParameter<bool>("printBlocks",false);

  rankByPt_ = 
    iConfig.getUntrackedParameter<bool>("rankByPt",false);

  entry_ = 0;


  LogDebug("PFEGammaCandidateChecker")
    <<" input collections : "<<inputTagPFCandidatesReco_<<" "<<inputTagPFCandidatesReReco_;
   
}



PFEGammaCandidateChecker::~PFEGammaCandidateChecker() { }



void 
PFEGammaCandidateChecker::beginRun(const edm::Run& run, 
			      const edm::EventSetup & es) { }


void 
PFEGammaCandidateChecker::analyze(const Event& iEvent, 
			     const EventSetup& iSetup) {
  
  LogDebug("PFEGammaCandidateChecker")<<"START event: "<<iEvent.id().event()
			 <<" in run "<<iEvent.id().run()<<endl;
  
  
  
  // get PFCandidates

  Handle<PFCandidateCollection> pfCandidatesReco;
  iEvent.getByLabel(inputTagPFCandidatesReco_, pfCandidatesReco);

  Handle<PFCandidateCollection> pfCandidatesReReco;
  iEvent.getByLabel(inputTagPFCandidatesReReco_, pfCandidatesReReco);

  Handle<PFJetCollection> pfJetsReco;
  iEvent.getByLabel(inputTagPFJetsReco_, pfJetsReco);

  Handle<PFJetCollection> pfJetsReReco;
  iEvent.getByLabel(inputTagPFJetsReReco_, pfJetsReReco);

  reco::PFCandidateCollection pfReco, pfReReco;  

  // to sort, one needs to copy
  if(rankByPt_)
    {
      pfReco=*pfCandidatesReco;
      pfReReco=*pfCandidatesReReco;
      sort(pfReco.begin(),pfReco.end(),greaterPt);
      sort(pfReReco.begin(),pfReReco.end(),greaterPt);
    }
  
  unsigned recoSize = pfReco.size();
  unsigned reRecoSize = pfReReco.size();
  //unsigned minSize = std::min(recoSize,pfReReco.size());
  bool differentCand = false;
  bool differentSize = pfReco.size() != pfReReco.size();
  if ( differentSize ) 
    std::cout << "+++WARNING+++ PFCandidate size changed for entry " 
	      << entry_ << " !" << endl
	      << " - RECO    size : " << pfReco.size() << endl 
	      << " - Re-RECO size : " << pfReReco.size() << endl;

  unsigned npr = 0;

  std::cout << "========= check pf -> ged =========" << std::endl;
  for( unsigned i=0; i<recoSize; i++ ) {
    
    const reco::PFCandidate & candReco = (rankByPt_) ? pfReco[i] : (*pfCandidatesReco)[i];
    const reco::PFCandidate * candReReco = NULL;

    switch( std::abs(candReco.pdgId()) ) {
    case 11:
      {
	std::cout << "got a pfElectron!" << std::endl;
	std::cout << '\t' << candReco << std::endl;
	std::cout << '\t' << candReco.gsfTrackRef()->ptMode() << std::endl;
	gsfequals findbygsf(candReco);
	reco::PFCandidateCollection::const_iterator found = 
	  std::find_if(pfReReco.begin(),pfReReco.end(),findbygsf);
	if( found != pfReReco.end() ) {
	  std::cout << "Found matching electron candidate by gsf track!" 
		    << std::endl;
	  candReReco = &*found;
	} else {
	  std::cout << "NO MATCHING GED ELECTRON" << std::endl;
	}
	reco::GsfElectronRef oldpf = candReco.gsfElectronRef();
	reco::GsfElectronRef pfeg;
	if( candReReco ) pfeg = candReReco->gsfElectronRef();
	if( oldpf.isNonnull() && pfeg.isNonnull() ) {
	  const double oldpf_sihih = oldpf->sigmaIetaIeta();
	  const double pfeg_sihih  = pfeg->sigmaIetaIeta();
	  std::cout << "matched set of gsf electrons" << std::endl;
	  if( oldpf_sihih < pfeg_sihih ) {
	    std::cout << "pfeg sigma-ieta-ieta > oldpf sigma-ieta-ieta" << std::endl;
	  }
	  if( oldpf_sihih > pfeg_sihih ) {
	    std::cout << "pfeg sigma-ieta-ieta < oldpf sigma-ieta-ieta" << std::endl;
	  }	  
	}	
      }
      break;
    case 22:
      { 
	scequals findbysc(candReco);
	reco::PFCandidateCollection::const_iterator found = 
	  std::find_if(pfReReco.begin(),pfReReco.end(),findbysc);
	if( candReco.photonRef().isNonnull() && found != pfReReco.end() ) {
	  std::cout << "Found matching photon candidate by parent!" 
		    << std::endl;
	  candReReco = &*found;
	}
      }
      break;
    default:
      break;
    }

    if( candReReco != NULL ) {
    
      double deltaE = (candReReco->energy()-candReco.energy())/(candReReco->energy()+candReco.energy());
      double deltaEta = candReReco->eta()-candReco.eta();
      double deltaPhi = candReReco->phi()-candReco.phi();
      if ( fabs(deltaE) > deltaEMax_ ||
	   fabs(deltaEta) > deltaEtaMax_ ||
	   fabs(deltaPhi) > deltaPhiMax_ ) { 
	differentCand = true;
	std::cout << "+++WARNING+++ PFCandidate (e or gamma) " << i 
		  << " changed  for entry " << entry_ << " ! " << std::endl 
		  << " - RECO     : " << candReco << std::endl
		  << " - Re-RECO  : " << *candReReco << std::endl
		  << " DeltaE   = : " << deltaE << std::endl
		  << " DeltaEta = : " << deltaEta << std::endl
		  << " DeltaPhi = : " << deltaPhi << std::endl << std::endl;
	if (printBlocks_) {
	  std::cout << "Elements in Block for RECO: " <<std::endl;
	  printElementsInBlocks(candReco);
	  std::cout << "Elements in Block for Re-RECO: " <<std::endl;
	  printElementsInBlocks(*candReReco);
	}
	if ( ++npr == 5 ) break;
      }
    }
  }
  std::cout << "========= check ged -> pf =========" << std::endl;
  for( unsigned i=0; i<reRecoSize; i++ ) {
    
    const reco::PFCandidate * candReco = NULL;
    const reco::PFCandidate & candReReco = (rankByPt_) ? pfReReco[i] : (*pfCandidatesReReco)[i];

    switch( std::abs(candReReco.pdgId()) ) {
    case 11:
      {
	std::cout << "got a ged electron!" << std::endl;
	std::cout << '\t' << candReReco << std::endl;
	std::cout << '\t' << candReReco.gsfTrackRef()->ptMode() << std::endl;
	gsfequals findbygsf(candReReco);
	reco::PFCandidateCollection::const_iterator found = 
	  std::find_if(pfReco.begin(),pfReco.end(),findbygsf);
	if( found != pfReco.end() ) {
	  std::cout << "Found matching electron candidate by gsf track!" 
		    << std::endl;
	  candReco = &*found;
	}
	reco::GsfElectronRef oldpf;
	reco::GsfElectronRef pfeg  = candReReco.gsfElectronRef();
	if( candReco ) oldpf = candReco->gsfElectronRef();
	if( pfeg.isNonnull() ) {
	  const double pfeg_sihih  = pfeg->sigmaIetaIeta();
	  if( ( std::abs(pfeg->superCluster()->eta()) < 1.479 && pfeg_sihih > 0.012 ) ||
	      ( std::abs(pfeg->superCluster()->eta()) > 1.479 && pfeg_sihih > 0.032 )   ) {
	    std::cout << "anomalously large sigma ieta ieta in pfeg!" << std::endl;
	    std::cout << "\tsihih    : " << pfeg_sihih << std::endl;
	    std::cout << "\te1x5     : " << pfeg->e1x5() << std::endl;
	    std::cout << "\te2x5 max : " << pfeg->e2x5Max() << std::endl;
	    std::cout << "\te5x5     : " << pfeg->e5x5() << std::endl;
	    std::cout << "\tfBrem    : " << pfeg->trackFbrem() << std::endl;
	  }
	  if( oldpf.isNonnull() && pfeg.isNonnull() ) {
	    const double oldpf_sihih = oldpf->sigmaIetaIeta();	    
	    std::cout << "matched set of gsf electrons" << std::endl;
	    if( oldpf_sihih < pfeg_sihih ) {
	      std::cout << "pfeg sigma-ieta-ieta > oldpf sigma-ieta-ieta" << std::endl;
	    }
	    if( oldpf_sihih > pfeg_sihih ) {
	      std::cout << "pfeg sigma-ieta-ieta < oldpf sigma-ieta-ieta" << std::endl;
	    }
	  }
	}	
      }
      break;
    case 22:
      { 
	scequals findbysc(candReReco);
	reco::PFCandidateCollection::const_iterator found = 
	  std::find_if(pfReco.begin(),pfReco.end(),findbysc);
	if( candReReco.photonRef().isNonnull() && found != pfReco.end() ) {
	  std::cout << "Found matching photon candidate by parent!" 
		    << std::endl;
	  candReco = &*found;
	}
      }
      break;
    default:
      break;
    }

    if( candReco != NULL ) {
    
      double deltaE = (candReReco.energy()-candReco->energy())/(candReReco.energy()+candReco->energy());
      double deltaEta = candReReco.eta()-candReco->eta();
      double deltaPhi = candReReco.phi()-candReco->phi();
      if ( fabs(deltaE) > deltaEMax_ ||
	   fabs(deltaEta) > deltaEtaMax_ ||
	   fabs(deltaPhi) > deltaPhiMax_ ) { 
	differentCand = true;
	std::cout << "+++WARNING+++ PFCandidate (e or gamma) " << i 
		  << " changed  for entry " << entry_ << " ! " << std::endl 
		  << " - RECO     : " << *candReco << std::endl
		  << " - Re-RECO  : " << candReReco << std::endl
		  << " DeltaE   = : " << deltaE << std::endl
		  << " DeltaEta = : " << deltaEta << std::endl
		  << " DeltaPhi = : " << deltaPhi << std::endl << std::endl;
	if (printBlocks_) {
	  std::cout << "Elements in Block for RECO: " <<std::endl;
	  printElementsInBlocks(*candReco);
	  std::cout << "Elements in Block for Re-RECO: " <<std::endl;
	  printElementsInBlocks(candReReco);
	}
	if ( ++npr == 5 ) break;
      }
    }    
  }


  if ( differentSize || differentCand ) { 
    printJets(*pfJetsReco, *pfJetsReReco);
    printMet(pfReco, pfReReco);
  }

  ++entry_;
  LogDebug("PFEGammaCandidateChecker")<<"STOP event: "<<iEvent.id().event()
			 <<" in run "<<iEvent.id().run()<<std::endl;
}


void PFEGammaCandidateChecker::printMet(const PFCandidateCollection& pfReco,
				  const PFCandidateCollection& pfReReco) const { 

  double metX = 0.;
  double metY = 0.;
  for( unsigned i=0; i<pfReco.size(); i++ ) {
    metX += pfReco[i].px();
    metY += pfReco[i].py();
  }
  double met = std::sqrt(metX*metX + metY*metY);
  std::cout << "MET RECO    = " << metX << " " << metY << " " << met << std::endl;

  metX = 0.;
  metY = 0.;
  for( unsigned i=0; i<pfReReco.size(); i++ ) {
    metX += pfReReco[i].px();
    metY += pfReReco[i].py();
  }
  met = std::sqrt(metX*metX + metY*metY);
  std::cout << "MET Re-RECO = " << metX << " " << metY << " " << met << std::endl;

}

void PFEGammaCandidateChecker::printJets(const PFJetCollection& pfJetsReco,
				   const PFJetCollection& pfJetsReReco) const { 

  bool differentSize = pfJetsReco.size() != pfJetsReReco.size();
  if ( differentSize ) 
    std::cout << "+++WARNING+++ PFJet size changed for entry " 
	      << entry_ << " !" << endl
	      << " - RECO    size : " << pfJetsReco.size() << endl 
	      << " - Re-RECO size : " << pfJetsReReco.size() << endl;
  unsigned minSize = pfJetsReco.size() < pfJetsReReco.size() ? pfJetsReco.size() : pfJetsReReco.size(); 
  unsigned npr = 0;
  for ( unsigned i = 0; i < minSize; ++i) {
    const reco::PFJet & candReco = pfJetsReco[i];
    const reco::PFJet & candReReco = pfJetsReReco[i];
    if ( candReco.et() < 20. && candReReco.et() < 20. ) break;
    double deltaE = (candReReco.et()-candReco.et())/(candReReco.et()+candReco.et());
    double deltaEta = candReReco.eta()-candReco.eta();
    double deltaPhi = candReReco.phi()-candReco.phi();
    if ( fabs(deltaE) > deltaEMax_ ||
	 fabs(deltaEta) > deltaEtaMax_ ||
	 fabs(deltaPhi) > deltaPhiMax_ ) { 
      std::cout << "+++WARNING+++ PFJet " << i 
		<< " changed  for entry " << entry_ << " ! " << std::endl 
		<< " - RECO     : " << candReco.et() << " " << candReco.eta() << " " << candReco.phi() << std::endl
		<< " - Re-RECO  : " << candReReco.et() << " " << candReReco.eta() << " " << candReReco.phi() << std::endl
		<< " DeltaE   = : " << deltaE << std::endl
		<< " DeltaEta = : " << deltaEta << std::endl
		<< " DeltaPhi = : " << deltaPhi << std::endl << std::endl;
      if ( ++npr == 5 ) break;
    } else { 
      std::cout << "Jet " << i << " " << candReco.et() << std::endl;
    }
  }

}


void PFEGammaCandidateChecker::printElementsInBlocks(const PFCandidate& cand,
						ostream& out) const {
  if(!out) return;

  PFBlockRef firstRef;

  assert(!cand.elementsInBlocks().empty() );
  for(unsigned i=0; i<cand.elementsInBlocks().size(); i++) {
    PFBlockRef blockRef = cand.elementsInBlocks()[i].first;

    if(blockRef.isNull()) {
      cerr<<"ERROR! no block ref!";
      continue;
    }

    if(!i) {
      out<<(*blockRef);
      firstRef = blockRef;
    }
    else if( blockRef!=firstRef) {
      cerr<<"WARNING! This PFCandidate is not made from a single block"<<endl;
    }
 
    out<<"\t"<<cand.elementsInBlocks()[i].second<<endl;
  }
}


