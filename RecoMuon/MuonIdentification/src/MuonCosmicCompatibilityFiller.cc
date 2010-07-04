// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      MuonCosmicCompatibilityFiller
//
//
// Original Author:  Adam Everett
// $Id: MuonCosmicCompatibilityFiller.cc,v 1.1 2010/06/18 23:13:22 aeverett Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"


#include "RecoMuon/MuonIdentification/interface/MuonCosmicCompatibilityFiller.h"
#include "RecoMuon/MuonIdentification/interface/MuonCosmicsId.h"

#include "DataFormats/MuonReco/interface/MuonCosmicCompatibility.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "TMath.h"


using namespace edm;

MuonCosmicCompatibilityFiller::MuonCosmicCompatibilityFiller(const edm::ParameterSet& iConfig):
  inputMuonCollection_(iConfig.getParameter<edm::InputTag>("InputMuonCollection")),
  inputTrackCollections_(iConfig.getParameter<std::vector<edm::InputTag> >("InputTrackCollections")),
  inputCosmicMuonCollection_(iConfig.getParameter<edm::InputTag>("InputCosmicMuonCollection")),
  inputVertexCollection_(iConfig.getParameter<edm::InputTag>("InputVertexCollection")),
  theService(0)
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

MuonCosmicCompatibilityFiller::~MuonCosmicCompatibilityFiller() {
  if (theService) delete theService;
}

reco::MuonCosmicCompatibility
MuonCosmicCompatibilityFiller::fillCompatibility( const reco::Muon& muon, edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  const std::string theCategory = "MuonCosmicCompatibilityFiller";

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
MuonCosmicCompatibilityFiller::cosmicTime(const reco::Muon& muon) const {
  
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
    
    if (nDThit ){
      if (nCSChit == 0 ) result *= 2;
      if (nCSChit > 0.5*nDThit ) result /= 2;
    } else { // no DTHits
      result = 0.;
    }
  }//muon time is valid
  
  return result;
} 

unsigned int
MuonCosmicCompatibilityFiller::backToBack2LegCosmic(const edm::Event& iEvent, const reco::Muon& muon) const {
  
  unsigned int result = 0;
  reco::TrackRef track;
  if ( muon.isGlobalMuon()  )            track = muon.innerTrack();
  else if ( muon.isTrackerMuon() )       track = muon.track();
  else if ( muon.isStandAloneMuon() )    return false;
  
  math::XYZPoint RefVtx;
  RefVtx.SetXYZ(0, 0, 0);
  
  edm::Handle<reco::VertexCollection> pvHandle;
  iEvent.getByLabel(inputVertexCollection_,pvHandle);
  const reco::VertexCollection & vertices = *pvHandle.product();
  for(reco::VertexCollection::const_iterator it=vertices.begin() ; it!=vertices.end() ; ++it){
    RefVtx = it->position();
  }
  
  if (track.isNonnull()) {
    for (unsigned int iColl = 0; iColl<inputTrackCollections_.size(); ++iColl){
      edm::Handle<reco::TrackCollection> trackHandle;
      iEvent.getByLabel(inputTrackCollections_[iColl],trackHandle);
      if( !trackHandle.failedToGet() ) {
	for(reco::TrackCollection::const_iterator iTrack = trackHandle->begin(); iTrack != trackHandle->end(); ++iTrack) {
	  //check if they share a point, set threshold to a given value or to d0error if larger
	  if ((iTrack->d0() == track->d0()) && (iTrack->d0Error() == track->d0Error())) {
	    //std::cout << "same track, skipping" <<  std::endl; 
	    continue;
	  }
	  const double ipErr = (double)track->d0Error();
	  double ipThreshold  = std::max(ipThreshold_, 3.*ipErr);
	  if (fabs(track->dxy(RefVtx) + iTrack->dxy(RefVtx)) < ipThreshold) {
	    std::pair<double, double> matchAngDPt (muonid::matchTracks(*iTrack, *track));
	    if ( matchAngDPt.first < angleThreshold_) {
	      if ( matchAngDPt.second < deltaPt_ )   {
		result++; //break;
	      }               
	    } // 
	  } // dxy match
	} // loop over tracks
      } // track collection valid
    } // loop over track collections
  } // muon has a track
  return result;
}

//
//Check overlap between collections, use shared hits info
//
bool MuonCosmicCompatibilityFiller::isOverlappingMuon(const edm::Event& iEvent, const reco::Muon& muon) const{
  
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
  iEvent.getByLabel(inputCosmicMuonCollection_, muonHandle);
  
  // Global Tracking Geometry
  //ESHandle<GlobalTrackingGeometry> trackingGeometry;
  //iSetup.get<GlobalTrackingGeometryRecord>().get(trackingGeometry);
  


  // PV
  math::XYZPoint RefVtx;
  RefVtx.SetXYZ(0, 0, 0);
  
  edm::Handle<reco::VertexCollection> pvHandle;
  iEvent.getByLabel(inputVertexCollection_,pvHandle);
  const reco::VertexCollection & vertices = *pvHandle.product();
  
  
  if( !muonHandle.failedToGet() ) {
    for ( reco::MuonCollection::const_iterator cosmicMuon = muonHandle->begin();cosmicMuon !=  muonHandle->end(); ++cosmicMuon ) {
      
      reco::TrackRef outertrack = muon.outerTrack();
      reco::TrackRef costrack = cosmicMuon->outerTrack();
      
      bool isUp = false;
      if( outertrack->phi() > 0 ) isUp = true; 
      
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

	for( trackingRecHit_iterator coshit = costrack->recHitsBegin(); coshit != costrack->recHitsEnd(); ++coshit ) {
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
          double ipThreshold  = std::max(ipThreshold_, 3.*ipErr);
	  for(reco::VertexCollection::const_iterator it=vertices.begin() ; it!=vertices.end() ; ++it) {
	    RefVtx = it->position();
	    if( fabs(outertrack->dxy(RefVtx) + costrack->dxy(RefVtx)) < ipThreshold ) isCloseIP = true; break;
	  }
	  if( !isCloseIP ) continue;

	  // step3
	  GlobalPoint muonRefVtx( outertrack->vx(), outertrack->vy(), outertrack->vz() );
	  GlobalPoint cosmicRefVtx( costrack->vx(), costrack->vy(), costrack->vz() );
	  float dist = (muonRefVtx - cosmicRefVtx).mag();
	  if( dist < 0.1 ) isCloseRef = true;
	  //if( !isCloseRef ) continue;

	  for( trackingRecHit_iterator trkhit = outertrack->recHitsBegin(); trkhit != outertrack->recHitsEnd(); ++trkhit ) {
	    if( (*trkhit)->isValid() ) {
	      for( trackingRecHit_iterator coshit = costrack->recHitsBegin(); coshit != costrack->recHitsEnd(); ++coshit ) {
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

unsigned int MuonCosmicCompatibilityFiller::pvMatches(const edm::Event& iEvent, const reco::Muon& muonTrack, bool isLoose) const {
  
  unsigned int result = 0;
  reco::TrackRef track;
  if ( muonTrack.isGlobalMuon()  )         track = muonTrack.innerTrack();
  else if ( muonTrack.isTrackerMuon() )    track = muonTrack.track();
  else if (muonTrack.isStandAloneMuon())   track = muonTrack.standAloneMuon();
  
  math::XYZPoint RefVtx;
  RefVtx.SetXYZ(0, 0, 0);
  
  edm::Handle<reco::VertexCollection> pvHandle;
  iEvent.getByLabel(inputVertexCollection_,pvHandle);
  const reco::VertexCollection & vertices = *pvHandle.product();
  for(reco::VertexCollection::const_iterator it=vertices.begin() ; it!=vertices.end() ; ++it){
    RefVtx = it->position();
    
    if (isLoose) { 
      if ( track.isNonnull() )  {
	if ( fabs( (*track).dxy(RefVtx) ) < maxdxyLoose_ && fabs( (*track).dz(RefVtx) ) < maxdzLoose_ )   result++;
      }
    } else {
      if ( track.isNonnull() )  {
	if ( fabs( (*track).dxy(RefVtx) ) < maxdxyTight_ &&  fabs( (*track).dz(RefVtx) ) < maxdzTight_ )   result++;
      }
    }
  }
  
  return result;
}

float MuonCosmicCompatibilityFiller::isGoodCosmic(const edm::Event& iEvent, const reco::Muon& muon, bool CheckMuonID ) const {
  
  // return >=1 = identify as cosmic muon (the more like cosmics, the higher is the number) 
  // return 0.0 = identify as collision muon
  
  //check cosmic phi, only certain sectors
  //reco::TrackRef outertrack = muon.outerTrack();
  //if( !(outertrack->phi() > 0.35 &&  outertrack->phi() < 2.79) && !(outertrack->phi() < -0.35 &&  outertrack->phi() > -2.79)) return 0.0;
  
  float result = 0;
  // simple decision tree
  if( muon.isGlobalMuon() ) {
    // short cut to reject cosmic event
    unsigned int nLoosePV = pvMatches(iEvent, muon, true );
    unsigned int nTightPV = pvMatches(iEvent, muon, false);

    unsigned int nOppTracks = backToBack2LegCosmic(iEvent,muon);

    if (nTightPV == 0) result += 0.5;
    if (nLoosePV == 0) result += 0.5;
    
    if (result > 0){
      if (inputTrackCollections_.size() == nOppTracks) result += nOppTracks;
      else if (nOppTracks > 0) result += 1;
    }

    float cosmicTimeCompat = cosmicTime(muon);
    result += cosmicTimeCompat ;
    
    bool isOverlapping = isOverlappingMuon(iEvent, muon);
    // this only makes sense if the overlap check is with a traversing muon. Otherwise, what's the point.
    if (nLoosePV == 0){
      if (isOverlapping) result += 1.5;
    } else {
      if (isOverlapping) result += 0.5;
    }

    if (result > 1 && CheckMuonID && isMuonID( muon )) result *=2; //extra bonus

  }
  
  return result;
}

bool MuonCosmicCompatibilityFiller::isMuonID( const reco::Muon& imuon ) const
{
  bool result = false;
  // initial set up using Jordan's study: GlobalMuonPromptTight + TMOneStationLoose
  if( muon::isGoodMuon(imuon, muon::GlobalMuonPromptTight) && muon::isGoodMuon(imuon, muon::TMOneStationLoose) ) result = true;
  
  return result;
}
