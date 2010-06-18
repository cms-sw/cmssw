// -*- C++ -*-
//
// Package:    GlobalTrackingTools
// Class:      GlobalCosmicCompatibilityFiller
//
//
// Original Author:  Adam Everett
// $Id: GlobalCosmicCompatibilityFiller.cc,v 1.6 2010/06/08 19:27:08 aeverett Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/MuonCosmicCompatibility.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"


#include "RecoMuon/GlobalTrackingTools/interface/GlobalCosmicCompatibilityFiller.h"

#include "DataFormats/MuonReco/interface/MuonCosmicCompatibility.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "TMath.h"


//#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
//#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
//#include "TrackingTools/PatternTools/interface/Trajectory.h"

//#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"


using namespace edm;

GlobalCosmicCompatibilityFiller::GlobalCosmicCompatibilityFiller(const edm::ParameterSet& iConfig):
  inputMuonCollection_(iConfig.getParameter<edm::InputTag>("InputMuonCollection")),inputTrackCollection_(iConfig.getParameter<edm::InputTag>("InputTrackCollection")),inputCosmicCollection_(iConfig.getParameter<edm::InputTag>("InputCosmicCollection")),inputVertexCollection_(iConfig.getParameter<edm::InputTag>("InputVertexCollection")),theService(0)
{
  // service parameters
  edm::ParameterSet serviceParameters = iConfig.getParameter<edm::ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters);     
  
  //ip and vertex
  maxdxyLoose_ = iConfig.getParameter<double>("maxdxyLoose"); 
  maxdzLoose_ = iConfig.getParameter<double>("maxdzLoose");
  maxdxyTight_ = iConfig.getParameter<double>("maxdxyTight");
  maxdzTight_ = iConfig.getParameter<double>("maxdzTight");
  minNDOF_ = iConfig.getParameter<double>("minNDOF");
  minvProb_ = iConfig.getParameter<double>("minvProb");
  ipThreshold_ = iConfig.getParameter<double>("ipCut");
  //kinematical vars
  deltaPhi_ = iConfig.getParameter<double>("deltaPhi");
  deltaPt_ = iConfig.getParameter<double>("deltaPt");
  angleThreshold_ = iConfig.getParameter<double>("angleCut"); 
  //time
  offTimePos_ = iConfig.getParameter<double>("offTimePos");
  offTimeNeg_ = iConfig.getParameter<double>("offTimeNeg");
  //rechits
  sharedHits_ = iConfig.getParameter<int>("sharedHits");
  sharedFrac_ = iConfig.getParameter<double>("sharedFrac");
  
}

GlobalCosmicCompatibilityFiller::~GlobalCosmicCompatibilityFiller() {
  if (theService) delete theService;
}

reco::MuonCosmicCompatibility
GlobalCosmicCompatibilityFiller::fillCompatibility( const reco::Muon& muon, edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  const std::string theCategory = "GlobalCosmicCompatibilityFiller";

  reco::MuonCosmicCompatibility returnComp;
  
  theService->update(iSetup);
  
  float timeCompatibility = cosmicTime(muon);
  float backToBackCompatibility = backToBack2LegCosmic(iEvent,muon);    
  float overlapCompatibility = isOverlappingMuon(iEvent,muon);
  
  returnComp.timeCompatibility = timeCompatibility;
  returnComp.backToBackCompatibility = backToBackCompatibility;
  returnComp.overlapCompatibility = overlapCompatibility;
  returnComp.cosmicCompatibility = isGoodCosmic(iEvent,muon,true);
  
  return returnComp;
  
}

float 
GlobalCosmicCompatibilityFiller::cosmicTime(const reco::Muon& muon) const {
  
  float result = 0.0;
  
  if( muon.isTimeValid() ) { 
    //currently cuts large negative times
    if ( muon.time().timeAtIpInOut < offTimeNeg_ || muon.time().timeAtIpInOut > offTimePos_) result = 1.0;
    
    // temporary: currently only DT timing is reliable
    int nDThit = 0;
    int nCSChit = 0;
    int nRPChit = 0;
    reco::TrackRef outertrack = muon.outerTrack();
    if( outertrack.isNonnull() ) {
      for( trackingRecHit_iterator trkhit = outertrack->recHitsBegin(); trkhit != outertrack->recHitsEnd(); trkhit++ ) {
	if( (*trkhit)->isValid() ) {
	  DetId recoid = (*trkhit)->geographicalId();
	  if( recoid.subdetId()  == MuonSubdetId::DT ) nDThit++;
	  if( recoid.subdetId()  == MuonSubdetId::CSC ) nCSChit++;
	  if( recoid.subdetId()  == MuonSubdetId::RPC ) nRPChit++;
	}
      }
    }
    
    if( nDThit == 0 ) result = 0.0;
    else {
      if( nCSChit / (double)(nCSChit+nDThit) > 0.1 ) result = 1.0;
    }
  }
  
  return result;
} 

bool 
GlobalCosmicCompatibilityFiller::backToBack2LegCosmic(const edm::Event& iEvent, const reco::Muon& muon) const {
  
  bool result = false;
  reco::TrackRef track;
  if ( muon.isGlobalMuon()  )            track = muon.innerTrack();
  else if ( muon.isTrackerMuon() )       track = muon.track();
  else if ( muon.isStandAloneMuon() )    return false;
  
  math::XYZPoint RefVtx;
  RefVtx.SetXYZ(0, 0, 0);
  
  edm::Handle<reco::VertexCollection> pvHandle;
  iEvent.getByLabel(inputVertexCollection_,pvHandle);
  const reco::VertexCollection & vertices = *pvHandle.product();
  for(reco::VertexCollection::const_iterator it=vertices.begin() ; it!=vertices.end() ; ++it)
    {
      RefVtx = it->position();
    }
  
  if (track.isNonnull()) {
    edm::Handle<reco::TrackCollection> trackHandle;
    iEvent.getByLabel(inputTrackCollection_,trackHandle);
    if( !trackHandle.failedToGet() ) {
      for(reco::TrackCollection::const_iterator iTrack = trackHandle->begin(); iTrack != trackHandle->end(); ++iTrack) {
	//check if they share a point, set threshold to a given value or to d0error if larger
	if ((iTrack->d0() == track->d0()) && (iTrack->d0Error() == track->d0Error())) {
	    //std::cout << "same track, skipping" <<  std::endl; 
	    continue;
	}
	const double ipErr = (double)track->d0Error();
	double ipThreshold  = max(ipThreshold_, ipErr);
	if (fabs(track->dxy(RefVtx) + iTrack->dxy(RefVtx)) > ipThreshold) {return false;} else {
	  if ( angleBetween(*track,*iTrack) < angleThreshold_) {return false;} else {
	    if ( fabs( track->pt() -  iTrack->pt())/fabs(max(track->pt(),iTrack->pt())) < deltaPt_ )   {result = true; break;}               
	  }
	}
      }
    }
  }
  
  return result;
}

//
//Check overlap between collections, use shared hits info
//
bool GlobalCosmicCompatibilityFiller::isOverlappingMuon(const edm::Event& iEvent, const reco::Muon& muon) const{
  
  // 4 steps in this module
  // step1 : check whether it's 1leg cosmic muon or not
  // step2 : both muons (muons and muonsFromCosmics1Leg) should have close IP
  // step3 : both muons should share very close reference point
  // step4 : check shared hits in both muon tracks

  // check if this muon is available in muonsFromCosmics collection
  bool overlappingMuon = false;
  if( !muon.isGlobalMuon() ) return false;
  
  // reco muons for cosmics
  edm::Handle<reco::MuonCollection> muonHandle;
  iEvent.getByLabel(inputCosmicCollection_, muonHandle);
  
  // Global Tracking Geometry
  //ESHandle<GlobalTrackingGeometry> trackingGeometry;
  //iSetup.get<GlobalTrackingGeometryRecord>().get(trackingGeometry);
  


  // PV
  math::XYZPoint RefVtx;
  RefVtx.SetXYZ(0, 0, 0);
  
  edm::Handle<reco::VertexCollection> pvHandle;
  iEvent.getByLabel(inputVertexCollection_,pvHandle);
  const reco::VertexCollection & vertices = *pvHandle.product();
  for(reco::VertexCollection::const_iterator it=vertices.begin() ; it!=vertices.end() ; ++it) {
      RefVtx = it->position();
  }
  
  
  if( !muonHandle.failedToGet() ) {
    for ( reco::MuonCollection::const_iterator cosmicMuon = muonHandle->begin();cosmicMuon !=  muonHandle->end(); ++cosmicMuon ) {
      
      //            std::cout << "before" << cosmicMuon->pt()<<" " << muon.pt() << std::endl;
//    if ( cosmicMuon->innerTrack() == muon.innerTrack() || cosmicMuon->outerTrack() == muon.outerTrack()) return true;
      
      reco::TrackRef outertrack = muon.outerTrack();
      reco::TrackRef costrack = cosmicMuon->outerTrack();
      
      bool isUp = false;
      if( outertrack->phi() > 0 ) isUp = true; 
      
      //	     std::cout << "of track = " << outertrack->phi() << " " << isUp << std::endl;
      // shared hits 
      int RecHitsMuon = outertrack->numberOfValidHits();
      int RecHitsCosmicMuon = 0;
      int shared = 0;
      // count hits for same hemisphere
      if( costrack.isNonnull() ) {
	int nhitsUp = 0;
	int nhitsDown = 0;
	bool isCosmic1Leg = false;
	bool isCloseIP = false;
	bool isCloseRef = false;

	for( trackingRecHit_iterator coshit = costrack->recHitsBegin(); coshit != costrack->recHitsEnd(); coshit++ ) {
	  if( (*coshit)->isValid() ) {
	    DetId id((*coshit)->geographicalId());
	    double hity = theService->trackingGeometry()->idToDet(id)->position().y();
	    if( hity > 0 ) nhitsUp++;
	    if( hity < 0 ) nhitsDown++;

	    if( isUp && hity > 0 ) RecHitsCosmicMuon++;
	    if( !isUp && hity < 0 ) RecHitsCosmicMuon++;
	  }
	}
	// step1
	if( nhitsUp > 0 && nhitsDown > 0 ) isCosmic1Leg = true;
	//if( !isCosmic1Leg ) continue;
	
	if( outertrack.isNonnull() ) {
	  // step2
          const double ipErr = (double)outertrack->d0Error();
          double ipThreshold  = max(ipThreshold_, ipErr);
	  if( fabs(outertrack->dxy(RefVtx) + costrack->dxy(RefVtx)) < ipThreshold ) isCloseIP = true;
	  if( !isCloseIP ) continue;

	  // step3
	  GlobalPoint muonRefVtx( outertrack->vx(), outertrack->vy(), outertrack->vz() );
	  GlobalPoint cosmicRefVtx( costrack->vx(), costrack->vy(), costrack->vz() );
	  float dist = (muonRefVtx - cosmicRefVtx).mag();
	  if( dist < 0.1 ) isCloseRef = true;
	  //if( !isCloseRef ) continue;

	  for( trackingRecHit_iterator trkhit = outertrack->recHitsBegin(); trkhit != outertrack->recHitsEnd(); trkhit++ ) {
	    if( (*trkhit)->isValid() ) {
	      for( trackingRecHit_iterator coshit = costrack->recHitsBegin(); coshit != costrack->recHitsEnd(); coshit++ ) {
		if( (*coshit)->isValid() ) {
		  if( (*trkhit)->geographicalId() == (*coshit)->geographicalId() ) {
		    if( ((*trkhit)->localPosition() - (*coshit)->localPosition()).mag()< 10e-5 ) shared++;
		  }
		  
		}
	      }
	    }
	  }
	}
      }
      // step4
      double fraction = -1;
      if( RecHitsMuon != 0 ) fraction = shared/(double)RecHitsMuon;
    // 	     std::cout << "shared = " << shared << " " << fraction << " " << RecHitsMuon << " " << RecHitsCosmicMuon << std::endl;
      if( shared > sharedHits_ && fraction > sharedFrac_ ) {
	overlappingMuon = true;
	break;
      }
    }
  }
  
  return overlappingMuon;
}

bool GlobalCosmicCompatibilityFiller::isCosmicVertex(const edm::Event& iEvent, const reco::Muon& muon) const
{
  
  bool result = true;
  
  edm::Handle<reco::VertexCollection> pvHandle;
  iEvent.getByLabel(inputVertexCollection_,pvHandle);
  const reco::VertexCollection & vertices = *pvHandle.product();
  // int count = 0; 
  for(reco::VertexCollection::const_iterator it=vertices.begin() ; it!=vertices.end() ; ++it)
    {
      //      count++;
      //	  std::cout << "count " << count << std::endl;
      if(it->ndof() < minNDOF_ || fabs(it->z()) > 20 || fabs(it->position().rho()) > 5  || 
	 TMath::Prob(it->chi2(),(int)it->ndof()) < minvProb_) return false;
      
    }
  
  return result;
}

bool GlobalCosmicCompatibilityFiller::isIpCosmic(const edm::Event& iEvent, const reco::Muon& muonTrack, bool isLoose) const {
  
  bool result = false;
  reco::TrackRef track;
  if ( muonTrack.isGlobalMuon()  )         track = muonTrack.innerTrack();
  else if ( muonTrack.isTrackerMuon() )    track = muonTrack.track();
  else if (muonTrack.isStandAloneMuon())   track = muonTrack.standAloneMuon();
  
  math::XYZPoint RefVtx;
  RefVtx.SetXYZ(0, 0, 0);
  
  edm::Handle<reco::VertexCollection> pvHandle;
  iEvent.getByLabel(inputVertexCollection_,pvHandle);
  const reco::VertexCollection & vertices = *pvHandle.product();
  for(reco::VertexCollection::const_iterator it=vertices.begin() ; it!=vertices.end() ; ++it)
    {
      RefVtx = it->position();
    }
  
  
  if (isLoose) { 
    if ( track.isNonnull() )  {
      if ( fabs( (*track).dxy(RefVtx) ) < maxdxyLoose_ ) {result = false;} else {
	if ( fabs( (*track).dz(RefVtx) ) > maxdzLoose_ )   result = true;
      }
    }
  } else {
    if ( track.isNonnull() )  {
      if ( fabs( (*track).dxy(RefVtx) ) < maxdxyTight_ ) {result = false;} else {
	if ( fabs( (*track).dz(RefVtx) ) > maxdzTight_ )   result = true;
      }
    }
  }
  
  return result;
}

float GlobalCosmicCompatibilityFiller::isGoodCosmic(const edm::Event& iEvent, const reco::Muon& muon, bool CheckMuonID ) const {
  
  // return 1.0 = identify as cosmic muon
  // return 0.0 = identify as collision muon
  
  //check cosmic phi, only certain sectors
  //reco::TrackRef outertrack = muon.outerTrack();
  //if( !(outertrack->phi() > 0.35 &&  outertrack->phi() < 2.79) && !(outertrack->phi() < -0.35 &&  outertrack->phi() > -2.79)) return 0.0;

  // simple decision tree
  if( muon.isGlobalMuon() ) {
    // short cut to reject cosmic event
    if (!isCosmicVertex(iEvent, muon)) return 0.0;
    else {
      // option: muon id cut
      if( CheckMuonID && !isMuonID( muon ) ) return 0.0;
      // IP?
      if (isIpCosmic(iEvent, muon,false)) {
	// if yes, check b-to-b
	if (backToBack2LegCosmic(iEvent,muon)) {
	  // if b-to-b = yes, it's cosmic!
	  return 1.0;
	}
	// if b-to-b = no, try muon timing! 
	else if( cosmicTime(muon) ) {
	  return 1.0;
	}
	else return 0.0;
      }
      else {
	// if IP = no, check the overlapping
	if (!isOverlappingMuon(iEvent,muon)) {
	  // okey let's check IP loose
	  if (isIpCosmic(iEvent, muon,true)) {
	    // okey let's check muon timing.
	    if( cosmicTime(muon) ) return 1.0;
	    else return 0.0;
	  }
	  else return 0.0;
	}
	else return 0.0;
      }
    }
  }
  
  return 0.0;
}

bool GlobalCosmicCompatibilityFiller::isMuonID( const reco::Muon& imuon ) const
{
  bool result = false;
  // initial set up using Jordan's study: GlobalMuonPromptTight + TMOneStationLoose
  if( muon::isGoodMuon(imuon, muon::GlobalMuonPromptTight) && muon::isGoodMuon(imuon, muon::TMOneStationLoose) ) result = true;
  
  return result;
}
