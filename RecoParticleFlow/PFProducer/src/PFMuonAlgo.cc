#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

bool
PFMuonAlgo::isMuon( const reco::PFBlockElement& elt ) {

  const reco::PFBlockElementTrack* eltTrack 
    = dynamic_cast<const reco::PFBlockElementTrack*>(&elt);

  assert ( eltTrack );
  reco::MuonRef muonRef = eltTrack->muonRef();
  
  return isMuon(muonRef);

}

bool
PFMuonAlgo::isLooseMuon( const reco::PFBlockElement& elt ) {

  const reco::PFBlockElementTrack* eltTrack 
    = dynamic_cast<const reco::PFBlockElementTrack*>(&elt);

  assert ( eltTrack );
  reco::MuonRef muonRef = eltTrack->muonRef();

  return isLooseMuon(muonRef);

}

bool
PFMuonAlgo::isMuon( const reco::MuonRef& muonRef ) {

  if ( !muonRef.isNonnull() ) return false;
  if ( !muonRef->isGlobalMuon() ) return false;
  // if ( !muonRef->isTrackerMuon() ) return false;
  if ( !muonRef->isStandAloneMuon() ) return false;

  reco::TrackRef standAloneMu = muonRef->standAloneMuon();
  reco::TrackRef combinedMu = muonRef->combinedMuon();
  reco::TrackRef trackerMu = muonRef->track();
 
  /*
  std::cout << " Global  Muon pt error " 
	    << combinedMu->ptError()/combinedMu->pt() << " " 
	    << combinedMu->pt() 
	    << std::endl; 
  std::cout << " Tracker Muon pt error " 
	    << trackerMu->ptError()/trackerMu->pt() << " " 
	    << trackerMu->pt() 
	    << std::endl;
  std::cout << " STAlone Muon pt error " 
	    << standAloneMu->ptError()/standAloneMu->pt() << " " 
	    << standAloneMu->pt() 
	    << std::endl; 
  */

  if ( muonRef->isTrackerMuon() ) { 
    double sigmaCombined = combinedMu->ptError()/(combinedMu->pt()*combinedMu->pt());
    double sigmaTracker = trackerMu->ptError()/(trackerMu->pt()*trackerMu->pt());
    double sigmaStandAlone = standAloneMu->ptError()/(standAloneMu->pt()*standAloneMu->pt());
    
    bool combined = combinedMu->ptError()/combinedMu->pt() < 0.20;
    bool tracker = trackerMu->ptError()/trackerMu->pt() < 0.20;
    bool standAlone = standAloneMu->ptError()/standAloneMu->pt() < 0.20; 
    
    double delta1 =  combined && tracker ?
      fabs(1./combinedMu->pt() -1./trackerMu->pt())
      /sqrt(sigmaCombined*sigmaCombined + sigmaTracker*sigmaTracker) : 100.; 
    double delta2 = combined && standAlone ?
      fabs(1./combinedMu->pt() -1./standAloneMu->pt())
      /sqrt(sigmaCombined*sigmaCombined + sigmaStandAlone*sigmaStandAlone) : 100.;
    double delta3 = standAlone && tracker ?
      fabs(1./standAloneMu->pt() -1./trackerMu->pt())
      /sqrt(sigmaStandAlone*sigmaStandAlone + sigmaTracker*sigmaTracker) : 100.;
    
    double delta =  
      standAloneMu->hitPattern().numberOfValidMuonDTHits()+
      standAloneMu->hitPattern().numberOfValidMuonCSCHits() > 0 ? 
      std::min(delta3,std::min(delta1,delta2)) : std::max(delta3,std::max(delta1,delta2));
    // std::cout << "delta = " << delta << std::endl;
    
    double ratio = 
      combinedMu->ptError()/combinedMu->pt()
      / (trackerMu->ptError()/trackerMu->pt());
    //if ( ratio > 2. && delta < 3. ) std::cout << "ALARM ! " << ratio << ", " << delta << std::endl;
    
    // Quality check on the hits in the muon chambers 
    // (at least two stations hit, or last station hit)
    bool quality =
      standAloneMu->hitPattern().numberOfValidMuonDTHits() > 12 ||
      standAloneMu->hitPattern().numberOfValidMuonCSCHits() > 6 ||
      muon::isGoodMuon(*muonRef,muon::TMLastStationLoose) ||
      muon::isGoodMuon(*muonRef,muon::TMLastStationOptimizedLowPtLoose);

    bool result =  ( combinedMu->pt() < 50. || ratio < 2. ) && delta < 3.;
    result = result && muon::isGoodMuon(*muonRef,muon::GlobalMuonPromptTight);
    /*
    if ( result && !quality ) 
      std::cout << " pt (STA/TRA) : " << standAloneMu->pt() 
		<< " +/- " << standAloneMu->ptError()/standAloneMu->pt() 
		<< " and " << trackerMu->pt() 
		<< " +/- " << trackerMu->ptError()/trackerMu->pt() 
		<< " and " << combinedMu->pt() 
		<< " +/- " << combinedMu->ptError()/combinedMu->pt() 
		<< " eta : " << standAloneMu->eta() << std::endl
		<< " delta/ratio = " << delta << "/" << ratio << std::endl
		<< " DT Hits : " << standAloneMu->hitPattern().numberOfValidMuonDTHits()
		<< "/" << standAloneMu->hitPattern().numberOfLostMuonDTHits()
		<< " CSC Hits : " << standAloneMu->hitPattern().numberOfValidMuonCSCHits()
		<< "/" << standAloneMu->hitPattern().numberOfLostMuonCSCHits()
		<< " RPC Hits : " << standAloneMu->hitPattern().numberOfValidMuonRPCHits()
		<< "/" << standAloneMu->hitPattern().numberOfLostMuonRPCHits() << std::endl
		<< " chi**2 STA : " << standAloneMu->normalizedChi2()
		<< " chi**2 GBL : " << combinedMu->normalizedChi2()
		<< std::endl 
		<< "TMLastStationLoose               "
		<< muon::isGoodMuon(*muonRef,muon::TMLastStationLoose) << std::endl       
		<< "TMLastStationTight               "
		<< muon::isGoodMuon(*muonRef,muon::TMLastStationTight) << std::endl    
		<< "TM2DCompatibilityLoose           "
		<< muon::isGoodMuon(*muonRef,muon::TM2DCompatibilityLoose) << std::endl 
		<< "TM2DCompatibilityTight           "
		<< muon::isGoodMuon(*muonRef,muon::TM2DCompatibilityTight) << std::endl
		<< "TMOneStationLoose                "
		<< muon::isGoodMuon(*muonRef,muon::TMOneStationLoose) << std::endl       
		<< "TMOneStationTight                "
		<< muon::isGoodMuon(*muonRef,muon::TMOneStationTight) << std::endl       
		<< "TMLastStationOptimizedLowPtLoose " 
		<< muon::isGoodMuon(*muonRef,muon::TMLastStationOptimizedLowPtLoose) << std::endl
		<< "TMLastStationOptimizedLowPtTight " 
		<< muon::isGoodMuon(*muonRef,muon::TMLastStationOptimizedLowPtTight) << std::endl 
		<< std::endl;

    */
    return result && quality;

  } else {
    // No tracker muon -> Request a perfect stand-alone muon, or an even better global muon
    bool result = false;
      
    // Check the quality of the stand-alone muon : 
    // good chi**2 and large number of hits and good pt error
    if ( ( standAloneMu->hitPattern().numberOfValidMuonDTHits() < 22 &&
	   standAloneMu->hitPattern().numberOfValidMuonCSCHits() < 15 ) ||
	 standAloneMu->normalizedChi2() > 10. || 
	 standAloneMu->ptError()/standAloneMu->pt() > 0.20 ) {
      result = false;
    } else { 
      // If the stand-alone muon is good, check the global muon
      if ( combinedMu->normalizedChi2() > standAloneMu->normalizedChi2() ) {
	// If the combined muon is worse than the stand-alone, it 
	// means that either the corresponding tracker track was not 
	// reconstructed, or that the sta muon comes from a late 
	// pion decay (hence with a momentum smaller than the track)
	// Take the stand-alone muon only if its momentum is larger
	// than that of the track
	result = standAloneMu->pt() > trackerMu->pt() ;
      } else { 
	// If the combined muon is better (and good enough), take the 
	// global muon
	result = 
	  combinedMu->ptError()/combinedMu->pt() < 
	  std::min(0.20,standAloneMu->ptError()/standAloneMu->pt());
      }
    }      
    /*
    if ( result ) 
      std::cout << " pt (STA/TRA) : " << standAloneMu->pt() 
		<< " +/- " << standAloneMu->ptError()/standAloneMu->pt() 
		<< " and " << trackerMu->pt() 
		<< " +/- " << trackerMu->ptError()/trackerMu->pt() 
		<< " and " << combinedMu->pt() 
		<< " +/- " << combinedMu->ptError()/combinedMu->pt() 
		<< " eta : " << standAloneMu->eta() << std::endl
		<< " DT Hits : " << standAloneMu->hitPattern().numberOfValidMuonDTHits()
		<< "/" << standAloneMu->hitPattern().numberOfLostMuonDTHits()
		<< " CSC Hits : " << standAloneMu->hitPattern().numberOfValidMuonCSCHits()
		<< "/" << standAloneMu->hitPattern().numberOfLostMuonCSCHits()
		<< " RPC Hits : " << standAloneMu->hitPattern().numberOfValidMuonRPCHits()
		<< "/" << standAloneMu->hitPattern().numberOfLostMuonRPCHits() << std::endl
		<< " chi**2 STA : " << standAloneMu->normalizedChi2()
		<< " chi**2 GBL : " << combinedMu->normalizedChi2()
		<< std::endl 
		<< "TMLastStationLoose               "
		<< muon::isGoodMuon(*muonRef,muon::TMLastStationLoose) << std::endl       
		<< "TMLastStationTight               "
		<< muon::isGoodMuon(*muonRef,muon::TMLastStationTight) << std::endl    
		<< "TM2DCompatibilityLoose           "
		<< muon::isGoodMuon(*muonRef,muon::TM2DCompatibilityLoose) << std::endl 
		<< "TM2DCompatibilityTight           "
		<< muon::isGoodMuon(*muonRef,muon::TM2DCompatibilityTight) << std::endl
		<< "TMOneStationLoose                "
		<< muon::isGoodMuon(*muonRef,muon::TMOneStationLoose) << std::endl       
		<< "TMOneStationTight                "
		<< muon::isGoodMuon(*muonRef,muon::TMOneStationTight) << std::endl       
		<< "TMLastStationOptimizedLowPtLoose " 
		<< muon::isGoodMuon(*muonRef,muon::TMLastStationOptimizedLowPtLoose) << std::endl
		<< "TMLastStationOptimizedLowPtTight " 
		<< muon::isGoodMuon(*muonRef,muon::TMLastStationOptimizedLowPtTight) << std::endl 
		<< std::endl;
    */
    return result;    
  }

  return false;

}

bool
PFMuonAlgo::isLooseMuon( const reco::MuonRef& muonRef ) {

  if ( !muonRef.isNonnull() ) return false;
  if ( !muonRef->isGlobalMuon() ) return false;
  if ( !muonRef->isTrackerMuon() ) return false;
  
  reco::TrackRef standAloneMu = muonRef->standAloneMuon();
  reco::TrackRef combinedMu = muonRef->combinedMuon();
  reco::TrackRef trackerMu = muonRef->track();

  // Some accuracy required on the momentum
  bool combined = combinedMu->ptError()/combinedMu->pt() < 0.25;
  bool tracker = trackerMu->ptError()/trackerMu->pt() < 0.25;
  // bool standAlone = standAloneMu->ptError()/standAloneMu->pt() < 0.25;

  bool combined40 = combinedMu->ptError()/combinedMu->pt() < 0.40;
  bool tracker40 = trackerMu->ptError()/trackerMu->pt() < 0.40;

  if ( !combined40 || !tracker40 ) return false;
  if ( !combined && !tracker ) return false;

  return true;

}

