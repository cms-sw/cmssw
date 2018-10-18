#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/MuonReco/interface/MuonCocktails.h"
#include <iostream>


using namespace std;
using namespace reco;
using namespace boost;

PFMuonAlgo::PFMuonAlgo() {
  pfCosmicsMuonCleanedCandidates_ = std::make_unique<reco::PFCandidateCollection>();
  pfCleanedTrackerAndGlobalMuonCandidates_= std::make_unique<reco::PFCandidateCollection>();
  pfFakeMuonCleanedCandidates_= std::make_unique<reco::PFCandidateCollection>();
  pfPunchThroughMuonCleanedCandidates_= std::make_unique<reco::PFCandidateCollection>();
  pfPunchThroughHadronCleanedCandidates_= std::make_unique<reco::PFCandidateCollection>();
  pfAddedMuonCandidates_ = std::make_unique<reco::PFCandidateCollection>();
  
}



void PFMuonAlgo::setParameters(const edm::ParameterSet& iConfig )
{

  if(iConfig.exists("maxDPtOPt"))
    maxDPtOPt_ = iConfig.getParameter<double>("maxDPtOPt");
  else
    maxDPtOPt_=1.0;

  if(iConfig.exists("minTrackerHits"))
    minTrackerHits_ = iConfig.getParameter<int>("minTrackerHits");
  else
    minTrackerHits_ = 8;

  if(iConfig.exists("minPixelHits"))
    minPixelHits_ = iConfig.getParameter<int>("minPixelHits");
  else
    minPixelHits_ = 1;

  if(iConfig.exists("trackQuality"))
    trackQuality_  = reco::TrackBase::qualityByName(iConfig.getParameter<std::string>("trackQuality"));
  else
    trackQuality_  = reco::TrackBase::qualityByName("highPurity");

  if(iConfig.exists("ptErrorScale"))
    errorCompScale_ = iConfig.getParameter<double>("ptErrorScale");
  else
    errorCompScale_ = 4.;

  if(iConfig.exists("eventFractionForCleaning"))
    eventFractionCleaning_ = iConfig.getParameter<double>("eventFractionForCleaning");
  else
    eventFractionCleaning_ = 0.75;

  if(iConfig.exists("dzPV"))
    dzPV_ = iConfig.getParameter<double>("dzPV");
  else
    dzPV_ = 0.2;

  if(iConfig.exists("postMuonCleaning"))
    postCleaning_ = iConfig.getParameter<bool>("postMuonCleaning");
  else
    postCleaning_ = false; //Disable by default (for HLT)

  if(iConfig.exists("minPtForPostCleaning"))
    minPostCleaningPt_ = iConfig.getParameter<double>("minPtForPostCleaning");
  else
    minPostCleaningPt_ = 20.;

  if(iConfig.exists("eventFactorForCosmics"))
    eventFactorCosmics_ = iConfig.getParameter<double>("eventFactorForCosmics");
  else
    eventFactorCosmics_ = 10.;


  if(iConfig.exists("metSignificanceForCleaning"))
    metSigForCleaning_ = iConfig.getParameter<double>("metSignificanceForCleaning");
  else
    metSigForCleaning_ = 3.;

  if(iConfig.exists("metSignificanceForRejection"))
    metSigForRejection_ = iConfig.getParameter<double>("metSignificanceForRejection");
  else
    metSigForRejection_ = 4.;

  if(iConfig.exists("metFactorForCleaning"))
    metFactorCleaning_ = iConfig.getParameter<double>("metFactorForCleaning");
  else
    metFactorCleaning_ = 4.;

  if(iConfig.exists("eventFractionForRejection"))
    eventFractionRejection_ = iConfig.getParameter<double>("eventFractionForRejection");
  else
    eventFractionRejection_ = 0.75;

  if(iConfig.exists("metFactorForRejection"))
    metFactorRejection_ = iConfig.getParameter<double>("metFactorForRejection");
  else
    metFactorRejection_ =4.;

  if(iConfig.exists("metFactorForHighEta"))
    metFactorHighEta_ = iConfig.getParameter<double>("metFactorForHighEta");
  else
    metFactorHighEta_=4;

  if(iConfig.exists("ptFactorForHighEta"))
    ptFactorHighEta_ = iConfig.getParameter<double>("ptFactorForHighEta");
  else
    ptFactorHighEta_ = 2.;

  if(iConfig.exists("metFactorForFakes"))
    metFactorFake_ = iConfig.getParameter<double>("metFactorForFakes");
  else
    metFactorFake_ = 4.;

  if(iConfig.exists("minMomentumForPunchThrough"))
    minPunchThroughMomentum_ = iConfig.getParameter<double>("minMomentumForPunchThrough");
  else
    minPunchThroughMomentum_=100.;

  if(iConfig.exists("minEnergyForPunchThrough"))
    minPunchThroughEnergy_ = iConfig.getParameter<double>("minEnergyForPunchThrough");
  else
    minPunchThroughEnergy_ = 100.;

  if(iConfig.exists("punchThroughFactor"))
    punchThroughFactor_ = iConfig.getParameter<double>("punchThroughFactor");
  else
    punchThroughFactor_ = 3.;

  if(iConfig.exists("punchThroughMETFactor"))
    punchThroughMETFactor_ = iConfig.getParameter<double>("punchThroughMETFactor");
  else
    punchThroughMETFactor_ = 4.;

  if(iConfig.exists("cosmicRejectionDistance"))
    cosmicRejDistance_ = iConfig.getParameter<double>("cosmicRejectionDistance");
  else
    cosmicRejDistance_ = 1.0;
}






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
PFMuonAlgo::isGlobalTightMuon( const reco::PFBlockElement& elt ) {

  const reco::PFBlockElementTrack* eltTrack 
    = dynamic_cast<const reco::PFBlockElementTrack*>(&elt);

  assert ( eltTrack );
  reco::MuonRef muonRef = eltTrack->muonRef();
  
  return isGlobalTightMuon(muonRef);

}

bool
PFMuonAlgo::isGlobalLooseMuon( const reco::PFBlockElement& elt ) {

  const reco::PFBlockElementTrack* eltTrack 
    = dynamic_cast<const reco::PFBlockElementTrack*>(&elt);

  assert ( eltTrack );
  reco::MuonRef muonRef = eltTrack->muonRef();

  return isGlobalLooseMuon(muonRef);

}

bool
PFMuonAlgo::isTrackerTightMuon( const reco::PFBlockElement& elt ) {

  const reco::PFBlockElementTrack* eltTrack 
    = dynamic_cast<const reco::PFBlockElementTrack*>(&elt);

  assert ( eltTrack );
  reco::MuonRef muonRef = eltTrack->muonRef();

  return isTrackerTightMuon(muonRef);

}

bool
PFMuonAlgo::isIsolatedMuon( const reco::PFBlockElement& elt ) {

  const reco::PFBlockElementTrack* eltTrack 
    = dynamic_cast<const reco::PFBlockElementTrack*>(&elt);

  assert ( eltTrack );
  reco::MuonRef muonRef = eltTrack->muonRef();

  return isIsolatedMuon(muonRef);

}

bool
PFMuonAlgo::isMuon(const reco::MuonRef& muonRef ){

  return isGlobalTightMuon(muonRef) || isTrackerTightMuon(muonRef) || isIsolatedMuon(muonRef);
}

bool
PFMuonAlgo::isLooseMuon(const reco::MuonRef& muonRef ){

  return (isGlobalLooseMuon(muonRef) || isTrackerLooseMuon(muonRef));

}

bool
PFMuonAlgo::isGlobalTightMuon( const reco::MuonRef& muonRef ) {

 if ( !muonRef.isNonnull() ) return false;

 if ( !muonRef->isGlobalMuon() ) return false;
 if ( !muonRef->isStandAloneMuon() ) return false;
 
 
 if ( muonRef->isTrackerMuon() ) { 
  
   bool result = muon::isGoodMuon(*muonRef,muon::GlobalMuonPromptTight);
   
   bool isTM2DCompatibilityTight =  muon::isGoodMuon(*muonRef,muon::TM2DCompatibilityTight);   
   int nMatches = muonRef->numberOfMatches();
   bool quality = nMatches > 2 || isTM2DCompatibilityTight;
   
   return result && quality;
   
 } else {

   reco::TrackRef standAloneMu = muonRef->standAloneMuon();
   
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
      
      reco::TrackRef combinedMu = muonRef->combinedMuon();
      reco::TrackRef trackerMu = muonRef->track();
            
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

    return result;    
  }

  return false;

}

bool
PFMuonAlgo::isTrackerTightMuon( const reco::MuonRef& muonRef ) {

  if ( !muonRef.isNonnull() ) return false;
    
  if(!muonRef->isTrackerMuon()) return false;
  
  reco::TrackRef trackerMu = muonRef->track();
  const reco::Track& track = *trackerMu;
  
  unsigned nTrackerHits =  track.hitPattern().numberOfValidTrackerHits();
  
  if(nTrackerHits<=12) return false;
  
  bool isAllArbitrated = muon::isGoodMuon(*muonRef,muon::AllArbitrated);
  
  bool isTM2DCompatibilityTight = muon::isGoodMuon(*muonRef,muon::TM2DCompatibilityTight);
  
  if(!isAllArbitrated || !isTM2DCompatibilityTight)  return false;

  if((trackerMu->ptError()/trackerMu->pt() > 0.10)){
    //std::cout<<" PT ERROR > 10 % "<< trackerMu->pt() <<std::endl;
    return false;
  }
  return true;
  
}

bool
PFMuonAlgo::isGlobalLooseMuon( const reco::MuonRef& muonRef ) {

  if ( !muonRef.isNonnull() ) return false;
  if ( !muonRef->isGlobalMuon() ) return false;
  if ( !muonRef->isStandAloneMuon() ) return false;
  
  reco::TrackRef standAloneMu = muonRef->standAloneMuon();
  reco::TrackRef combinedMu = muonRef->combinedMuon();
  reco::TrackRef trackerMu = muonRef->track();
 
  unsigned nMuonHits =
    standAloneMu->hitPattern().numberOfValidMuonDTHits() +
    2*standAloneMu->hitPattern().numberOfValidMuonCSCHits();
    
  bool quality = false;
  
  if ( muonRef->isTrackerMuon() ){

    bool result = combinedMu->normalizedChi2() < 100.;
    
    bool laststation =
      muon::isGoodMuon(*muonRef,muon::TMLastStationAngTight);
        
    int nMatches = muonRef->numberOfMatches();    
    
    quality = laststation && nMuonHits > 12 && nMatches > 1;

    return result && quality;
    
  }
  else{

    // Check the quality of the stand-alone muon : 
    // good chi**2 and large number of hits and good pt error
    if (  nMuonHits <=15  ||
	  standAloneMu->normalizedChi2() > 10. || 
	  standAloneMu->ptError()/standAloneMu->pt() > 0.20 ) {
      quality = false;
    }
   else { 
      // If the stand-alone muon is good, check the global muon
      if ( combinedMu->normalizedChi2() > standAloneMu->normalizedChi2() ) {
	// If the combined muon is worse than the stand-alone, it 
	// means that either the corresponding tracker track was not 
	// reconstructed, or that the sta muon comes from a late 
	// pion decay (hence with a momentum smaller than the track)
	// Take the stand-alone muon only if its momentum is larger
	// than that of the track

	// Note that here we even take the standAlone if it has a smaller pT, in contrast to GlobalTight
	if(standAloneMu->pt() > trackerMu->pt() || combinedMu->normalizedChi2()<5.) quality =  true;
      }
      else { 
	// If the combined muon is better (and good enough), take the 
	// global muon
	if(combinedMu->ptError()/combinedMu->pt() < std::min(0.20,standAloneMu->ptError()/standAloneMu->pt())) 
	  quality = true;
	
      }
   }         
  }
  

  return quality;

}


bool
PFMuonAlgo::isTrackerLooseMuon( const reco::MuonRef& muonRef ) {

  if ( !muonRef.isNonnull() ) return false;
  if(!muonRef->isTrackerMuon()) return false;

  reco::TrackRef trackerMu = muonRef->track();

  if(trackerMu->ptError()/trackerMu->pt() > 0.20) return false;

  // this doesn't seem to be necessary on the small samples looked at, but keep it around as insurance
  if(trackerMu->pt()>20.) return false;
    
  bool isAllArbitrated = muon::isGoodMuon(*muonRef,muon::AllArbitrated);
  bool isTMLastStationAngTight = muon::isGoodMuon(*muonRef,muon::TMLastStationAngTight);

  bool quality = isAllArbitrated && isTMLastStationAngTight;

  return quality;
  
}

bool
PFMuonAlgo::isIsolatedMuon( const reco::MuonRef& muonRef ){


  if ( !muonRef.isNonnull() ) return false;
  if ( !muonRef->isIsolationValid() ) return false;
  
  // Isolated Muons which are missed by standard cuts are nearly always global+tracker
  if ( !muonRef->isGlobalMuon() ) return false;

  // If it's not a tracker muon, only take it if there are valid muon hits

  reco::TrackRef standAloneMu = muonRef->standAloneMuon();

  if ( !muonRef->isTrackerMuon() ){
    if(standAloneMu->hitPattern().numberOfValidMuonDTHits() == 0 &&
       standAloneMu->hitPattern().numberOfValidMuonCSCHits() ==0) return false;
  }
  
  // for isolation, take the smallest pt available to reject fakes

  reco::TrackRef combinedMu = muonRef->combinedMuon();
  double smallestMuPt = combinedMu->pt();
  
  if(standAloneMu->pt()<smallestMuPt) smallestMuPt = standAloneMu->pt();
  
  if(muonRef->isTrackerMuon())
    {
      reco::TrackRef trackerMu = muonRef->track();
      if(trackerMu->pt() < smallestMuPt) smallestMuPt= trackerMu->pt();
    }

   
  double sumPtR03 = muonRef->isolationR03().sumPt;
  
  double relIso = sumPtR03/smallestMuPt;



  
  if(relIso<0.1) return true;
  else return false;
}

bool 
PFMuonAlgo::isTightMuonPOG(const reco::MuonRef& muonRef) {

  if(!muon::isGoodMuon(*muonRef,muon::GlobalMuonPromptTight)) return false;

  if(!muonRef->isTrackerMuon()) return false;
  
  if(muonRef->numberOfMatches()<2) return false;
  
  //const reco::TrackRef& combinedMuon = muonRef->combinedMuon();    
  const reco::TrackRef& combinedMuon = muonRef->globalTrack();    
  
  if(combinedMuon->hitPattern().numberOfValidTrackerHits()<11) return false;
  
  if(combinedMuon->hitPattern().numberOfValidPixelHits()==0) return false;
  
  if(combinedMuon->hitPattern().numberOfValidMuonHits()==0) return false;  

  return true;

}


bool 
PFMuonAlgo::hasValidTrack(const reco::MuonRef& muonRef,bool loose) {
  if(loose)
    return !muonTracks(muonRef).empty();
  else
    return !goodMuonTracks(muonRef).empty();

}


void 
PFMuonAlgo::printMuonProperties(const reco::MuonRef& muonRef){
  
  if ( !muonRef.isNonnull() ) return;
  
  bool isGL = muonRef->isGlobalMuon();
  bool isTR = muonRef->isTrackerMuon();
  bool isST = muonRef->isStandAloneMuon();

  std::cout<<" GL: "<<isGL<<" TR: "<<isTR<<" ST: "<<isST<<std::endl;
  std::cout<<" nMatches "<<muonRef->numberOfMatches()<<std::endl;
  
  if ( muonRef->isGlobalMuon() ){
    reco::TrackRef combinedMu = muonRef->combinedMuon();
    std::cout<<" GL,  pt: " << combinedMu->pt() 
	<< " +/- " << combinedMu->ptError()/combinedMu->pt() 
	     << " chi**2 GBL : " << combinedMu->normalizedChi2()<<std::endl;
    std::cout<< " Total Muon Hits : " << combinedMu->hitPattern().numberOfValidMuonHits()
	     << "/" << combinedMu->hitPattern().numberOfLostMuonHits()
	<< " DT Hits : " << combinedMu->hitPattern().numberOfValidMuonDTHits()
	<< "/" << combinedMu->hitPattern().numberOfLostMuonDTHits()
	<< " CSC Hits : " << combinedMu->hitPattern().numberOfValidMuonCSCHits()
	<< "/" << combinedMu->hitPattern().numberOfLostMuonCSCHits()
	<< " RPC Hits : " << combinedMu->hitPattern().numberOfValidMuonRPCHits()
	     << "/" << combinedMu->hitPattern().numberOfLostMuonRPCHits()<<std::endl;

    std::cout<<"  # of Valid Tracker Hits "<<combinedMu->hitPattern().numberOfValidTrackerHits()<<std::endl;
    std::cout<<"  # of Valid Pixel Hits "<<combinedMu->hitPattern().numberOfValidPixelHits()<<std::endl;
  }
  if ( muonRef->isStandAloneMuon() ){
    reco::TrackRef standAloneMu = muonRef->standAloneMuon();
    std::cout<<" ST,  pt: " << standAloneMu->pt() 
	<< " +/- " << standAloneMu->ptError()/standAloneMu->pt() 
	<< " eta : " << standAloneMu->eta()  
	<< " DT Hits : " << standAloneMu->hitPattern().numberOfValidMuonDTHits()
	<< "/" << standAloneMu->hitPattern().numberOfLostMuonDTHits()
	<< " CSC Hits : " << standAloneMu->hitPattern().numberOfValidMuonCSCHits()
	<< "/" << standAloneMu->hitPattern().numberOfLostMuonCSCHits()
	<< " RPC Hits : " << standAloneMu->hitPattern().numberOfValidMuonRPCHits()
	<< "/" << standAloneMu->hitPattern().numberOfLostMuonRPCHits()
	     << " chi**2 STA : " << standAloneMu->normalizedChi2()<<std::endl;
      }


  if ( muonRef->isTrackerMuon() ){
    reco::TrackRef trackerMu = muonRef->track();
    const reco::Track& track = *trackerMu;
    std::cout<<" TR,  pt: " << trackerMu->pt() 
	<< " +/- " << trackerMu->ptError()/trackerMu->pt() 
	     << " chi**2 TR : " << trackerMu->normalizedChi2()<<std::endl;    
    std::cout<<" nTrackerHits "<<track.hitPattern().numberOfValidTrackerHits()<<std::endl;
    std::cout<< "TMLastStationAngLoose               "
	<< muon::isGoodMuon(*muonRef,muon::TMLastStationAngLoose) << std::endl       
	<< "TMLastStationAngTight               "
	<< muon::isGoodMuon(*muonRef,muon::TMLastStationAngTight) << std::endl          
	<< "TMLastStationLoose               "
	<< muon::isGoodMuon(*muonRef,muon::TMLastStationLoose) << std::endl       
	<< "TMLastStationTight               "
	<< muon::isGoodMuon(*muonRef,muon::TMLastStationTight) << std::endl          
	<< "TMOneStationLoose                "
	<< muon::isGoodMuon(*muonRef,muon::TMOneStationLoose) << std::endl       
	<< "TMOneStationTight                "
	<< muon::isGoodMuon(*muonRef,muon::TMOneStationTight) << std::endl       
	<< "TMLastStationOptimizedLowPtLoose " 
	<< muon::isGoodMuon(*muonRef,muon::TMLastStationOptimizedLowPtLoose) << std::endl
	<< "TMLastStationOptimizedLowPtTight " 
	<< muon::isGoodMuon(*muonRef,muon::TMLastStationOptimizedLowPtTight) << std::endl 
	<< "TMLastStationOptimizedBarrelLowPtLoose " 
	<< muon::isGoodMuon(*muonRef,muon::TMLastStationOptimizedBarrelLowPtLoose) << std::endl
	<< "TMLastStationOptimizedBarrelLowPtTight " 
	<< muon::isGoodMuon(*muonRef,muon::TMLastStationOptimizedBarrelLowPtTight) << std::endl 
	<< std::endl;

  }

  std::cout<< "TM2DCompatibilityLoose           "
      << muon::isGoodMuon(*muonRef,muon::TM2DCompatibilityLoose) << std::endl 
      << "TM2DCompatibilityTight           "
	   << muon::isGoodMuon(*muonRef,muon::TM2DCompatibilityTight) << std::endl;



  if (	    muonRef->isGlobalMuon() &&  muonRef->isTrackerMuon() &&  muonRef->isStandAloneMuon() ){
    reco::TrackRef combinedMu = muonRef->combinedMuon();
    reco::TrackRef trackerMu = muonRef->track();
    reco::TrackRef standAloneMu = muonRef->standAloneMuon();
    
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

    std::cout << "delta = " << delta << " delta1 "<<delta1<<" delta2 "<<delta2<<" delta3 "<<delta3<<std::endl; 	 
    
    double ratio = 	 
      combinedMu->ptError()/combinedMu->pt() 	 
      / (trackerMu->ptError()/trackerMu->pt()); 	 
    //if ( ratio > 2. && delta < 3. ) std::cout << "ALARM ! " << ratio << ", " << delta << std::endl;
    std::cout<<" ratio "<<ratio<<" combined mu pt "<<combinedMu->pt()<<std::endl;
    //bool quality3 =  ( combinedMu->pt() < 50. || ratio < 2. ) && delta <  3.;


  }

    double sumPtR03 = muonRef->isolationR03().sumPt;
    double emEtR03 = muonRef->isolationR03().emEt;
    double hadEtR03 = muonRef->isolationR03().hadEt;    
    double relIsoR03 = (sumPtR03 + emEtR03 + hadEtR03)/muonRef->pt();
    double sumPtR05 = muonRef->isolationR05().sumPt;
    double emEtR05 = muonRef->isolationR05().emEt;
    double hadEtR05 = muonRef->isolationR05().hadEt;    
    double relIsoR05 = (sumPtR05 + emEtR05 + hadEtR05)/muonRef->pt();
    std::cout<<" 0.3 Radion Rel Iso: "<<relIsoR03<<" sumPt "<<sumPtR03<<" emEt "<<emEtR03<<" hadEt "<<hadEtR03<<std::endl;
    std::cout<<" 0.5 Radion Rel Iso: "<<relIsoR05<<" sumPt "<<sumPtR05<<" emEt "<<emEtR05<<" hadEt "<<hadEtR05<<std::endl;
  return;

}



std::vector<reco::Muon::MuonTrackTypePair> PFMuonAlgo::goodMuonTracks(const reco::MuonRef& muon,bool includeSA) {
  return muonTracks(muon,includeSA,maxDPtOPt_);
}



std::vector<reco::Muon::MuonTrackTypePair> PFMuonAlgo::muonTracks(const reco::MuonRef& muon,bool includeSA,double dpt) {



  std::vector<reco::Muon::MuonTrackTypePair> out;

  
  if(muon->globalTrack().isNonnull() && muon->globalTrack()->pt()>0) 
    if(muon->globalTrack()->ptError()/muon->globalTrack()->pt()<dpt)
      out.emplace_back(muon->globalTrack(),reco::Muon::CombinedTrack);

  if(muon->innerTrack().isNonnull() && muon->innerTrack()->pt()>0) 
    if(muon->innerTrack()->ptError()/muon->innerTrack()->pt()<dpt)//Here Loose!@
      out.emplace_back(muon->innerTrack(),reco::Muon::InnerTrack);

  bool pickyExists=false; 
  double pickyDpt=99999.; 
  if(muon->pickyTrack().isNonnull() && muon->pickyTrack()->pt()>0) {
    pickyDpt = muon->pickyTrack()->ptError()/muon->pickyTrack()->pt(); 
    if(pickyDpt<dpt) 
      out.emplace_back(muon->pickyTrack(),reco::Muon::Picky);
    pickyExists=true;
  }

  bool dytExists=false; 
  double dytDpt=99999.; 
  if(muon->dytTrack().isNonnull() && muon->dytTrack()->pt()>0) {
    dytDpt = muon->dytTrack()->ptError()/muon->dytTrack()->pt(); 
    if(dytDpt<dpt) 
      out.emplace_back(muon->dytTrack(),reco::Muon::DYT);
    dytExists=true;
  }

  //Magic: TPFMS is not a really good track especially under misalignment
  //IT is kind of crap because if mu system is displaced it can make a change
  //So allow TPFMS if there is no picky or the error of tpfms is better than picky
  //AND if there is no DYT or the error of tpfms is better than DYT
  if(muon->tpfmsTrack().isNonnull() && muon->tpfmsTrack()->pt()>0) { 
    double tpfmsDpt = muon->tpfmsTrack()->ptError()/muon->tpfmsTrack()->pt(); 
    if( ( (pickyExists && tpfmsDpt<pickyDpt) || (!pickyExists) ) && 
	( (dytExists   && tpfmsDpt<dytDpt)   || (!dytExists)   ) && 
	tpfmsDpt<dpt )
      out.emplace_back(muon->tpfmsTrack(),reco::Muon::TPFMS);
  }

  if(includeSA && muon->outerTrack().isNonnull())
    if(muon->outerTrack()->ptError()/muon->outerTrack()->pt()<dpt)
      out.emplace_back(muon->outerTrack(),reco::Muon::OuterTrack);

  return out;

}





////////////////////////////////////////////////////////////////////////////////////




bool PFMuonAlgo::reconstructMuon(reco::PFCandidate& candidate, const reco::MuonRef& muon, bool allowLoose) {
    using namespace std;
    using namespace reco;

    if (!muon.isNonnull())
      return false;


    

    bool isMu=false;

    if(allowLoose) 
      isMu = isMuon(muon) || isLooseMuon(muon);
    else
      isMu = isMuon(muon);

    if( !isMu)
      return false;





    //get the valid tracks(without standalone except we allow loose muons)
    //MIKE: Here we need to be careful. If we have a muon inside a dense 
    //jet environment often the track is badly measured. In this case 
    //we should not apply Dpt/Pt<1
 
    std::vector<reco::Muon::MuonTrackTypePair> validTracks = goodMuonTracks(muon);
    if (!allowLoose)
      validTracks = goodMuonTracks(muon);
    else
      validTracks = muonTracks(muon);

    if( validTracks.empty())
      return false;




    //check what is the track used.Rerun TuneP
    reco::Muon::MuonTrackTypePair bestTrackPair = muon::tevOptimized(*muon);
    
    TrackRef bestTrack = bestTrackPair.first;
    MuonTrackType trackType = bestTrackPair.second;



    MuonTrackTypePair trackPairWithSmallestError = getTrackWithSmallestError(validTracks);
    TrackRef trackWithSmallestError = trackPairWithSmallestError.first;
    
    if( trackType == reco::Muon::InnerTrack && 
	(!bestTrack->quality(trackQuality_) ||
	 bestTrack->ptError()/bestTrack->pt()> errorCompScale_*trackWithSmallestError->ptError()/trackWithSmallestError->pt() )) {
      bestTrack = trackWithSmallestError;
      trackType = trackPairWithSmallestError.second;
    }
    else if (trackType != reco::Muon::InnerTrack &&
	     bestTrack->ptError()/bestTrack->pt()> errorCompScale_*trackWithSmallestError->ptError()/trackWithSmallestError->pt())  {
      bestTrack = trackWithSmallestError;
      trackType = trackPairWithSmallestError.second;
      
    }


    changeTrack(candidate,std::make_pair(bestTrack,trackType));
    candidate.setMuonRef( muon );

    return true;
}




void  PFMuonAlgo::changeTrack(reco::PFCandidate& candidate,const MuonTrackTypePair& track) {
  using namespace reco;
    reco::TrackRef      bestTrack = track.first;
    MuonTrackType trackType = track.second;
    //OK Now redefine the canddiate with that track
    double px = bestTrack->px();
    double py = bestTrack->py();
    double pz = bestTrack->pz();
    double energy = sqrt(bestTrack->p()*bestTrack->p() + 0.1057*0.1057);

    candidate.setCharge(bestTrack->charge()>0 ? 1 : -1);
    candidate.setP4(math::XYZTLorentzVector(px,py,pz,energy));
    candidate.setParticleType(reco::PFCandidate::mu);
    //    candidate.setTrackRef( bestTrack );  
    candidate.setMuonTrackType(trackType);
    if(trackType == reco::Muon::InnerTrack)
      candidate.setVertexSource( PFCandidate::kTrkMuonVertex );
    else if(trackType == reco::Muon::CombinedTrack)
      candidate.setVertexSource( PFCandidate::kComMuonVertex );
    else if(trackType == reco::Muon::TPFMS)
      candidate.setVertexSource( PFCandidate::kTPFMSMuonVertex );
    else if(trackType == reco::Muon::Picky)
      candidate.setVertexSource( PFCandidate::kPickyMuonVertex );
    else if(trackType == reco::Muon::DYT)
      candidate.setVertexSource( PFCandidate::kDYTMuonVertex );
  }


reco::Muon::MuonTrackTypePair 
PFMuonAlgo::getTrackWithSmallestError(const std::vector<reco::Muon::MuonTrackTypePair>& tracks) {
    TrackPtErrorSorter sorter;
    return *std::min_element(tracks.begin(),tracks.end(),sorter);
}





void PFMuonAlgo::estimateEventQuantities(const reco::PFCandidateCollection* pfc)
{
  //SUM ET 
  sumetPU_ = 0.0;
  METX_=0.;
  METY_=0.;
  sumet_=0.0;
  for(reco::PFCandidateCollection::const_iterator i = pfc->begin();i!=pfc->end();++i) {
    sumet_+=i->pt();
    METX_+=i->px();
    METY_+=i->py();
  }

}





void PFMuonAlgo::postClean(reco::PFCandidateCollection*  cands) {
  using namespace std;
  using namespace reco;
  if (!postCleaning_)
    return;

  //Initialize vectors

  if(pfCosmicsMuonCleanedCandidates_.get() )
    pfCosmicsMuonCleanedCandidates_->clear();
  else 
    pfCosmicsMuonCleanedCandidates_.reset( new reco::PFCandidateCollection );

  if(pfCleanedTrackerAndGlobalMuonCandidates_.get() )
    pfCleanedTrackerAndGlobalMuonCandidates_->clear();
  else 
    pfCleanedTrackerAndGlobalMuonCandidates_.reset( new reco::PFCandidateCollection );

  if( pfFakeMuonCleanedCandidates_.get() )
     pfFakeMuonCleanedCandidates_->clear();
  else 
     pfFakeMuonCleanedCandidates_.reset( new reco::PFCandidateCollection );


  if( pfPunchThroughMuonCleanedCandidates_.get() )
     pfPunchThroughMuonCleanedCandidates_->clear();
  else 
     pfPunchThroughMuonCleanedCandidates_.reset( new reco::PFCandidateCollection );

  if( pfPunchThroughHadronCleanedCandidates_.get() )
     pfPunchThroughHadronCleanedCandidates_->clear();
  else 
     pfPunchThroughHadronCleanedCandidates_.reset( new reco::PFCandidateCollection );



  pfPunchThroughHadronCleanedCandidates_->clear();
  
  maskedIndices_.clear();

  //Estimate MET and SumET

  estimateEventQuantities(cands);

  
  std::vector<int> muons;
  std::vector<int> cosmics;
  //get the muons
  for(unsigned int i=0;i<cands->size();++i)  {
    const reco::PFCandidate& cand = (*cands)[i];
    if ( cand.particleId() == reco::PFCandidate::mu )
      muons.push_back(i);
      
  }
  //Then sort the muon indicess by decsending pt

  IndexPtComparator comparator(cands);
  std::sort(muons.begin(),muons.end(),comparator);


  //first kill cosmics
  double METXCosmics=0;
  double METYCosmics=0;
  double SUMETCosmics=0.0;

  for(unsigned int i=0;i<muons.size();++i) {
    const PFCandidate& pfc = cands->at(muons[i]);
    double origin=0.0;
    if(!vertices_->empty()&& vertices_->at(0).isValid() && ! vertices_->at(0).isFake())
      origin = pfc.muonRef()->muonBestTrack()->dxy(vertices_->at(0).position());

    if( origin> cosmicRejDistance_) {
      cosmics.push_back(muons[i]);
      METXCosmics +=pfc.px();
      METYCosmics +=pfc.py();
      SUMETCosmics +=pfc.pt();
    }
  }
  double MET2Cosmics = METXCosmics*METXCosmics+METYCosmics*METYCosmics;

  if ( SUMETCosmics > (sumet_-sumetPU_)/eventFactorCosmics_ && MET2Cosmics < METX_*METX_+ METY_*METY_)
    for(unsigned int i=0;i<cosmics.size();++i) {
      maskedIndices_.push_back(cosmics[i]);
      pfCosmicsMuonCleanedCandidates_->push_back(cands->at(cosmics[i]));
    }

  //Loop on the muons candidates and clean
  for(unsigned int i=0;i<muons.size();++i) {  
    if( cleanMismeasured(cands->at(muons[i]),muons[i]))
      continue;
    cleanPunchThroughAndFakes(cands->at(muons[i]),cands,muons[i]);
  
  }



  //OK Now do the hard job ->remove the candidates that were cleaned 
  removeDeadCandidates(cands,maskedIndices_);




}

void PFMuonAlgo::addMissingMuons(edm::Handle<reco::MuonCollection> muons, reco::PFCandidateCollection* cands) {
  if(!postCleaning_)
    return;


  if( pfAddedMuonCandidates_.get() )
     pfAddedMuonCandidates_->clear();
  else 
     pfAddedMuonCandidates_.reset( new reco::PFCandidateCollection );



  for ( unsigned imu = 0; imu < muons->size(); ++imu ) {
    reco::MuonRef muonRef( muons, imu );
    bool used = false;
    bool hadron = false;
    for(unsigned i=0; i<cands->size(); i++) {
      const PFCandidate& pfc = cands->at(i);
      if ( !pfc.trackRef().isNonnull() ) continue;
      if ( pfc.trackRef().isNonnull() && pfc.trackRef() == muonRef->track() )
	hadron = true;
      if ( !pfc.muonRef().isNonnull() ) continue;

      if ( pfc.muonRef()->innerTrack() == muonRef->innerTrack())
	used = true;
      else {
	// Check if the stand-alone muon is not a spurious copy of an existing muon 
	// (Protection needed for HLT)
	if ( pfc.muonRef()->isStandAloneMuon() && muonRef->isStandAloneMuon() ) { 
	  double dEta = pfc.muonRef()->standAloneMuon()->eta() - muonRef->standAloneMuon()->eta();
	  double dPhi = pfc.muonRef()->standAloneMuon()->phi() - muonRef->standAloneMuon()->phi();
	  double dR = sqrt(dEta*dEta + dPhi*dPhi);
	  if ( dR < 0.005 ) { 
	    used = true;
	  }
	}
      }
  
    if ( used ) break; 
    }

    if ( used ||hadron||(!muonRef.isNonnull()) ) continue;


    TrackMETComparator comparator(METX_,METY_);
    //Low pt dont need to be cleaned
  
    std::vector<reco::Muon::MuonTrackTypePair> tracks  = goodMuonTracks(muonRef,true);
    //If there is at least 1 track choice  try to change the track 
    if(!tracks.empty()) {

    //Find tracks that change dramatically MET or Pt
    std::vector<reco::Muon::MuonTrackTypePair> tracksThatChangeMET = tracksPointingAtMET(tracks);
    //From those tracks get the one with smallest MET 
    if (!tracksThatChangeMET.empty()) {
      reco::Muon::MuonTrackTypePair bestTrackType = *std::min_element(tracksThatChangeMET.begin(),tracksThatChangeMET.end(),comparator);

      //Make sure it is not cosmic
      if((vertices_->empty()) ||bestTrackType.first->dz(vertices_->at(0).position())<cosmicRejDistance_){
	
	//make a pfcandidate
	int charge = bestTrackType.first->charge()>0 ? 1 : -1;
	math::XYZTLorentzVector momentum(bestTrackType.first->px(),
					 bestTrackType.first->py(),
					 bestTrackType.first->pz(),
				       sqrt(bestTrackType.first->p()*bestTrackType.first->p()+0.1057*0.1057));
      
	cands->push_back( PFCandidate( charge, 
				      momentum,
				      reco::PFCandidate::mu ) );

	changeTrack(cands->back(),bestTrackType);

	if (muonRef->track().isNonnull() ) 
	  cands->back().setTrackRef( muonRef->track() );

	cands->back().setMuonRef(muonRef);


	pfAddedMuonCandidates_->push_back(cands->back());

      }
    }
    }
  }
}

std::pair<double,double> 
PFMuonAlgo::getMinMaxMET2(const reco::PFCandidate&pfc) {
  std::vector<reco::Muon::MuonTrackTypePair> tracks  = goodMuonTracks((pfc.muonRef()),true);

  double METXNO = METX_-pfc.px();
  double METYNO = METY_-pfc.py();
  std::vector<double> met2;
  for (unsigned int i=0;i<tracks.size();++i) {
    met2.push_back(pow(METXNO+tracks.at(i).first->px(),2)+pow(METYNO+tracks.at(i).first->py(),2));
  }

  //PROTECT for cases of only one track. If there is only one track it will crash .
  //Has never happened but could likely happen!

  if(tracks.size()>1)
    return std::make_pair(*std::min_element(met2.begin(),met2.end()),*std::max_element(met2.begin(),met2.end()));
  else
    return std::make_pair(0,0);
}


bool PFMuonAlgo::cleanMismeasured(reco::PFCandidate& pfc,unsigned int i ){
  using namespace std;
  using namespace reco;
  bool cleaned=false;

  //First define the MET without this guy
  double METNOX = METX_ - pfc.px();
  double METNOY = METY_ - pfc.py();
  double SUMETNO = sumet_  -pfc.pt(); 
  
  TrackMETComparator comparator(METNOX,METNOY);
  //Low pt dont need to be cleaned
  if (pfc.pt()<minPostCleaningPt_)
    return false;
  std::vector<reco::Muon::MuonTrackTypePair> tracks  = goodMuonTracks(pfc.muonRef(),false);
  


  //If there is more than 1 track choice  try to change the track 
  if(tracks.size()>1) {
    //Find tracks that change dramatically MET or Pt
    std::vector<reco::Muon::MuonTrackTypePair> tracksThatChangeMET = tracksWithBetterMET(tracks,pfc);
    //From those tracks get the one with smallest MET 
    if (!tracksThatChangeMET.empty()) {
      reco::Muon::MuonTrackTypePair bestTrackType = *std::min_element(tracksThatChangeMET.begin(),tracksThatChangeMET.end(),comparator);
      changeTrack(pfc,bestTrackType);

      pfCleanedTrackerAndGlobalMuonCandidates_->push_back(pfc);
      //update eventquantities
      METX_ = METNOX+pfc.px();
      METY_ = METNOY+pfc.py();
      sumet_=SUMETNO+pfc.pt();

    }      
  }

  //Now attempt to kill it 
  if (!(pfc.muonRef()->isGlobalMuon() && pfc.muonRef()->isTrackerMuon())) {
    //define MET significance and SUM ET
    double MET2 = METX_*METX_+METY_*METY_;
    double newMET2 = METNOX*METNOX+METNOY*METNOY;
    double METSig = sqrt(MET2)/sqrt(sumet_-sumetPU_);
    if( METSig>metSigForRejection_)
      if((newMET2 < MET2/metFactorRejection_) &&
	 ((SUMETNO-sumetPU_)/(sumet_-sumetPU_)<eventFractionRejection_)) {
	   pfFakeMuonCleanedCandidates_->push_back(pfc);
	   maskedIndices_.push_back(i);
	   METX_ = METNOX;
	   METY_ = METNOY;
	   sumet_=SUMETNO;
	   cleaned=true;
    }
    
  }
    return cleaned;

}
      
std::vector<reco::Muon::MuonTrackTypePair> 
PFMuonAlgo::tracksWithBetterMET(const std::vector<reco::Muon::MuonTrackTypePair>& tracks ,const reco::PFCandidate& pfc) {
  std::vector<reco::Muon::MuonTrackTypePair> outputTracks;

  double METNOX  = METX_ - pfc.px();
  double METNOY  = METY_ - pfc.py();
  double SUMETNO = sumet_  -pfc.pt(); 
  double MET2 = METX_*METX_+METY_*METY_;
  double newMET2=0.0;
  double newSUMET=0.0;
  double METSIG = sqrt(MET2)/sqrt(sumet_-sumetPU_);


  if(METSIG>metSigForCleaning_)
  for( unsigned int i=0;i<tracks.size();++i) {
    //calculate new SUM ET and MET2
    newSUMET = SUMETNO+tracks.at(i).first->pt()-sumetPU_;
    newMET2  = pow(METNOX+tracks.at(i).first->px(),2)+pow(METNOY+tracks.at(i).first->py(),2);
    
    if(newSUMET/(sumet_-sumetPU_)>eventFractionCleaning_ &&  newMET2<MET2/metFactorCleaning_)
      outputTracks.push_back(tracks.at(i));
  }
  
  
  return outputTracks;
}


std::vector<reco::Muon::MuonTrackTypePair> 
PFMuonAlgo::tracksPointingAtMET(const std::vector<reco::Muon::MuonTrackTypePair>& tracks) {
  std::vector<reco::Muon::MuonTrackTypePair> outputTracks;


  double newMET2=0.0;

  for( unsigned int i=0;i<tracks.size();++i) {
    //calculate new SUM ET and MET2
    newMET2  = pow(METX_+tracks.at(i).first->px(),2)+pow(METY_+tracks.at(i).first->py(),2);
    
    if(newMET2<(METX_*METX_+METY_*METY_)/metFactorCleaning_)
      outputTracks.push_back(tracks.at(i));
  }
  
  
  return outputTracks;
}
  

  
void PFMuonAlgo::setInputsForCleaning(const reco::VertexCollection*  vertices) {
  vertices_ = vertices;
}

bool PFMuonAlgo::cleanPunchThroughAndFakes(reco::PFCandidate&pfc,reco::PFCandidateCollection* cands,unsigned int imu ){
  using namespace reco;

  bool cleaned=false;

  if (pfc.pt()<minPostCleaningPt_)
    return false;


  double METXNO = METX_-pfc.pt();
  double METYNO = METY_-pfc.pt();
  double MET2NO = METXNO*METXNO+METYNO*METYNO;
  double MET2   = METX_*METX_+METY_*METY_;
  bool fake1=false;

  std::pair<double,double> met2 = getMinMaxMET2(pfc);

  //Check for Fakes at high pseudorapidity
  if(pfc.muonRef()->standAloneMuon().isNonnull()) 
    fake1 =fabs ( pfc.eta() ) > 2.15 && 
      met2.first<met2.second/2 &&
      MET2NO < MET2/metFactorHighEta_ && 
      pfc.muonRef()->standAloneMuon()->pt() < pfc.pt()/ptFactorHighEta_;

  double factor = std::max(2.,2000./(sumet_-pfc.pt()-sumetPU_));
  bool fake2 = ( pfc.pt()/(sumet_-sumetPU_) < 0.25 && MET2NO < MET2/metFactorFake_ && met2.first<met2.second/factor );

  bool punchthrough =pfc.p() > minPunchThroughMomentum_ &&  
    pfc.rawHcalEnergy() > minPunchThroughEnergy_ && 
    pfc.rawEcalEnergy()+pfc.rawHcalEnergy() > pfc.p()/punchThroughFactor_ &&
    !isIsolatedMuon(pfc.muonRef()) && MET2NO < MET2/punchThroughMETFactor_;


  if(fake1 || fake2||punchthrough) {
    // Find the block of the muon
    const PFCandidate::ElementsInBlocks& eleInBlocks = pfc.elementsInBlocks();
    if ( !eleInBlocks.empty() ) { 
      PFBlockRef blockRefMuon = eleInBlocks[0].first;
      unsigned indexMuon = eleInBlocks[0].second;
      if (eleInBlocks.size()>1)
	indexMuon = eleInBlocks[1].second;
	  
      // Check if the muon gave rise to a neutral hadron
      double iHad = 1E9;
      bool hadron = false;
      for ( unsigned i = imu+1; i < cands->size(); ++i ) { 
	const PFCandidate& pfcn = cands->at(i);
	    const PFCandidate::ElementsInBlocks& ele = pfcn.elementsInBlocks();
	    if ( ele.empty() ) { 
	      continue;
	    }
	    PFBlockRef blockRefHadron = ele[0].first;
	    unsigned indexHadron = ele[0].second;
	    // We are out of the block -> exit the loop
	    if ( blockRefHadron.key() != blockRefMuon.key() ) break;
	    // Check that this particle is a neutral hadron
	    if ( indexHadron == indexMuon && 
		 pfcn.particleId() == reco::PFCandidate::h0 ) {
	      iHad = i;
	      hadron = true;
	    }
	    if ( hadron ) break;
	  }
	  
	  if ( hadron ) { 

    double rescaleFactor = cands->at(iHad).p()/cands->at(imu).p();
	    METX_ -=  cands->at(imu).px() + cands->at(iHad).px();
	    METY_ -=  cands->at(imu).py() + cands->at(iHad).py();
	    sumet_ -=cands->at(imu).pt();
	    cands->at(imu).rescaleMomentum(rescaleFactor);
	    maskedIndices_.push_back(iHad);
	    pfPunchThroughHadronCleanedCandidates_->push_back(cands->at(iHad));
	    cands->at(imu).setParticleType(reco::PFCandidate::h);
	    pfPunchThroughMuonCleanedCandidates_->push_back(cands->at(imu));
	    METX_ +=  cands->at(imu).px();
	    METY_ +=  cands->at(imu).py();	  
	    sumet_ += cands->at(imu).pt();

	  } else if ( fake1 || fake2 ) {
	    METX_ -=  cands->at(imu).px();
	    METY_ -=  cands->at(imu).py();	  
	    sumet_ -= cands->at(imu).pt();
	    maskedIndices_.push_back(imu);
	    pfFakeMuonCleanedCandidates_->push_back(cands->at(imu));
	    cleaned=true;
	  }
    }
  }
  return cleaned;
}


void  PFMuonAlgo::removeDeadCandidates(reco::PFCandidateCollection* obj, const std::vector<unsigned int>& indices)
{
  size_t N = indices.size();
  size_t collSize = obj->size();

  for (size_t i = 0 ; i < N ; ++i)
    obj->at(indices.at(i)) = obj->at(collSize-i-1);

  obj->resize(collSize - indices.size());
}
