/**
 *  Class: GlobalMuonRefitter
 *
 *  Description:
 *
 *
 *  $Date: 2009/02/23 09:55:34 $
 *  $Revision: 1.6 $
 *
 *  Authors :
 *  P. Traczyk, SINS Warsaw
 *
 **/

#include "RecoMuon/GlobalTrackingTools/interface/GlobalMuonRefitter.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "TrackingTools/TrackFitters/interface/RecHitLessByDet.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"


#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

using namespace std;
using namespace edm;

//----------------
// Constructors --
//----------------

GlobalMuonRefitter::GlobalMuonRefitter(const edm::ParameterSet& par,
				       const MuonServiceProxy* service) : 
  theDTRecHitLabel(par.getParameter<InputTag>("DTRecSegmentLabel")),
  theCSCRecHitLabel(par.getParameter<InputTag>("CSCRecSegmentLabel")),
  theService(service) {

  theCategory = par.getUntrackedParameter<string>("Category", "Muon|RecoMuon|GlobalMuon|GlobalMuonRefitter");

  theHitThreshold = par.getParameter<int>("HitThreshold");
  theDTChi2Cut  = par.getParameter<double>("Chi2CutDT");
  theCSCChi2Cut = par.getParameter<double>("Chi2CutCSC");
  theRPCChi2Cut = par.getParameter<double>("Chi2CutRPC");

  // Refit direction
  string refitDirectionName = par.getParameter<string>("RefitDirection");
  
  if (refitDirectionName == "insideOut" ) theRefitDirection = insideOut;
  else if (refitDirectionName == "outsideIn" ) theRefitDirection = outsideIn;
  else 
    throw cms::Exception("TrackTransformer constructor") 
      <<"Wrong refit direction chosen in TrackTransformer ParameterSet"
      << "\n"
      << "Possible choices are:"
      << "\n"
      << "RefitDirection = insideOut or RefitDirection = outsideIn";
  
  theFitterName = par.getParameter<string>("Fitter");  
  thePropagatorName = par.getParameter<string>("Propagator");

  theSkipStation        = par.getParameter<int>("SkipStation");
  theTrackerSkipSystem	= par.getParameter<int>("TrackerSkipSystem");
  theTrackerSkipSection	= par.getParameter<int>("TrackerSkipSection");//layer, wheel, or disk depending on the system

  theTrackerRecHitBuilderName = par.getParameter<string>("TrackerRecHitBuilder");
  theMuonRecHitBuilderName = par.getParameter<string>("MuonRecHitBuilder");

  theRPCInTheFit = par.getParameter<bool>("RefitRPCHits");

  theCacheId_TC = theCacheId_GTG = theCacheId_MG = theCacheId_TRH = 0;

}

//--------------
// Destructor --
//--------------

GlobalMuonRefitter::~GlobalMuonRefitter() {
}


//
// set Event
//
void GlobalMuonRefitter::setEvent(const edm::Event& event) {

  event.getByLabel(theDTRecHitLabel, theDTRecHits);
  event.getByLabel(theCSCRecHitLabel, theCSCRecHits);
}


void GlobalMuonRefitter::setServices(const EventSetup& setup) {

  theService->eventSetup().get<TrajectoryFitter::Record>().get(theFitterName,theFitter);

  // Transient Rechit Builders
  unsigned long long newCacheId_TRH = setup.get<TransientRecHitRecord>().cacheIdentifier();
  if ( newCacheId_TRH != theCacheId_TRH ) {
    LogDebug(theCategory) << "TransientRecHitRecord changed!";
    setup.get<TransientRecHitRecord>().get(theTrackerRecHitBuilderName,theTrackerRecHitBuilder);
    setup.get<TransientRecHitRecord>().get(theMuonRecHitBuilderName,theMuonRecHitBuilder);
  }
}


//
// build a combined tracker-muon trajectory
//
vector<Trajectory> GlobalMuonRefitter::refit(const reco::Track& globalTrack, 
                   const int theMuonHitsOption) const {

  // MuonHitsOption: 0 - tracker only
  //                 1 - include all muon hits
  //                 2 - include only first muon hit(s)
  //                 3 - include only selected muon hits

  vector<int> stationHits(4,0);

  ConstRecHitContainer allRecHits; // all muon rechits
  ConstRecHitContainer allRecHitsTemp; // all muon rechits temp
  ConstRecHitContainer fmsRecHits; // only first muon rechits
  ConstRecHitContainer selectedRecHits; // selected muon rechits

  LogTrace(theCategory) << " *** GlobalMuonRefitter *** option " << theMuonHitsOption << endl;

  LogTrace(theCategory) << " Track momentum before refit: " << globalTrack.pt() << endl;

  reco::TransientTrack track(globalTrack,&*(theService->magneticField()),theService->trackingGeometry());

  for (trackingRecHit_iterator hit = track.recHitsBegin(); hit != track.recHitsEnd(); ++hit)
    if((*hit)->isValid())
      if ( (*hit)->geographicalId().det() == DetId::Tracker )
	allRecHitsTemp.push_back(theTrackerRecHitBuilder->build(&**hit));
      else if ( (*hit)->geographicalId().det() == DetId::Muon ){
	if( (*hit)->geographicalId().subdetId() == 3 && !theRPCInTheFit){
	  LogTrace(theCategory) << "RPC Rec Hit discarged"; 
	  continue;
	}
	allRecHitsTemp.push_back(theMuonRecHitBuilder->build(&**hit));
      }

  allRecHits = getRidOfSelectStationHits(allRecHitsTemp);  
  //    printHits(allRecHits);
  LogTrace(theCategory) << " Hits size: " << allRecHits.size() << endl;

  vector <Trajectory> outputTraj;

  if ((theMuonHitsOption == 1) || (theMuonHitsOption == 3)) {
    // refit the full track with all muon hits
    vector <Trajectory> globalTraj = transform(globalTrack, track, allRecHits);

    if (!globalTraj.size()) {
      LogTrace(theCategory) << "No trajectory from the TrackTransformer!" << endl;
      return vector<Trajectory>();
    }

    LogTrace(theCategory) << " Initial trajectory state: " 
                          << globalTraj.front().lastMeasurement().updatedState().freeState()->parameters() << endl;
  
    if (theMuonHitsOption == 1 )
      outputTraj.push_back(globalTraj.front());
    
    if (theMuonHitsOption == 3 ) { 
      checkMuonHits(globalTrack, allRecHits, stationHits);
      selectedRecHits = selectMuonHits(globalTraj.front(),stationHits);
      LogTrace(theCategory) << " Selected hits size: " << selectedRecHits.size() << endl;  
      outputTraj = transform(globalTrack, track, selectedRecHits);
    }     
  } else if (theMuonHitsOption == 2 )  {
      getFirstHits(globalTrack, allRecHits, fmsRecHits);
      outputTraj = transform(globalTrack, track, fmsRecHits);
    } 

  if (outputTraj.size()) {
    LogTrace(theCategory) << "Refitted pt: " << outputTraj.front().firstMeasurement().updatedState().globalParameters().momentum().perp() << endl;
    return outputTraj;
  } else {
    LogTrace(theCategory) << "No refitted Tracks... " << endl;
    return vector<Trajectory>();
  }
  
}


//
//
//
void GlobalMuonRefitter::checkMuonHits(const reco::Track& muon, 
				       ConstRecHitContainer& all,
				       std::vector<int>& hits) const {

  LogTrace(theCategory) << " GlobalMuonRefitter::checkMuonHits " << endl;

  float coneSize = 20.0;
  int dethits[4];
  for ( int i=0; i<4; i++ ) hits[i]=dethits[i]=0;

  // loop through all muon hits and calculate the maximum # of hits in each chamber
  for (ConstRecHitContainer::const_iterator imrh = all.begin(); imrh != all.end(); imrh++ ) {
        
    if ( (*imrh != 0 ) && !(*imrh)->isValid() ) continue;
  
    int station = 0;
    int detRecHits = 0;
    MuonRecHitContainer dRecHits;
      
    DetId id = (*imrh)->geographicalId();

    // Skip tracker hits
    if (id.det()!=DetId::Muon) continue;

    if ( id.subdetId() == MuonSubdetId::DT ) {
      DTChamberId did(id.rawId());
      DTLayerId lid(id.rawId());
      station = did.station();

      // Get the 1d DT RechHits from this layer
      DTRecHitCollection::range dRecHits = theDTRecHits->get(lid);

      for (DTRecHitCollection::const_iterator ir = dRecHits.first; ir != dRecHits.second; ir++ ) {
	double rhitDistance = fabs(ir->localPosition().x()-(**imrh).localPosition().x());
	if ( rhitDistance < coneSize ) detRecHits++;
        LogTrace(theCategory)	<< "       " << (ir)->localPosition() << "  " << (**imrh).localPosition()
               << " Distance: " << rhitDistance << " recHits: " << detRecHits << endl;
      }
    }// end of if DT
    else if ( id.subdetId() == MuonSubdetId::CSC ) {
    
      CSCDetId did(id.rawId());
      station = did.station();

      // Get the CSC Rechits from this layer
      CSCRecHit2DCollection::range dRecHits = theCSCRecHits->get(did);      

      for (CSCRecHit2DCollection::const_iterator ir = dRecHits.first; ir != dRecHits.second; ir++ ) {
	double rhitDistance = (ir->localPosition()-(**imrh).localPosition()).mag();
	if ( rhitDistance < coneSize ) detRecHits++;
        LogTrace(theCategory)	<< ir->localPosition() << "  " << (**imrh).localPosition()
	       << " Distance: " << rhitDistance << " recHits: " << detRecHits << endl;
      }
    }
    else {
      if ( id.subdetId() != MuonSubdetId::RPC ) LogError(theCategory)<<" Wrong Hit Type ";
      continue;      
    }
      
    if ( (station > 0) && (station < 5) ) {
      if ( detRecHits > hits[station-1] ) hits[station-1] = detRecHits;
    }

  } // end of loop over muon rechits

  for ( int i = 0; i < 4; i++ ) 
    LogTrace(theCategory) <<" Station "<<i+1<<": "<<hits[i]<<" "<<dethits[i] <<endl; 

  LogTrace(theCategory) << "CheckMuonHits: "<<all.size();
  
  // check order of muon measurements
  if ( (all.size() > 1) &&
       ( all.front()->globalPosition().mag() >
	 all.back()->globalPosition().mag() ) ) {
    LogTrace(theCategory)<< "reverse order: ";
    stable_sort(all.begin(),all.end(),RecHitLessByDet(alongMomentum));
  }
}


//
// Get the hits from the first muon station (containing hits)
//
void GlobalMuonRefitter::getFirstHits(const reco::Track& muon, 
				       ConstRecHitContainer& all,
				       ConstRecHitContainer& first) const {

  LogTrace(theCategory) << " GlobalMuonRefitter::getFirstHits " << endl;

  // check order of muon measurements
  if ( (all.size() > 1) &&
       ( all.front()->globalPosition().mag() >
	 all.back()->globalPosition().mag() ) ) {
    LogTrace(theCategory)<< "reverse order: ";
    stable_sort(all.begin(),all.end(),RecHitLessByDet(alongMomentum));
  }
  
  int station1 = -999;
  int station2 = -999;
  for (ConstRecHitContainer::const_iterator ihit = all.begin(); ihit != all.end(); ihit++ ) {

    if ( !(*ihit)->isValid() ) continue;
    station1 = -999; station2 = -999;
    // store muon hits one at a time.
    first.push_back(*ihit);
    DetId id = (*ihit)->geographicalId();

    // Skip tracker hits
    if (id.det()!=DetId::Muon) continue;
    
    ConstMuonRecHitPointer immrh = dynamic_cast<const MuonTransientTrackingRecHit*>((*ihit).get()); //FIXME
    
    // get station of 1st hit if it is in DT
    if ( (*immrh).isDT()  ) {
      DTChamberId did(id.rawId());
      station1 = did.station();
    }
    // otherwise get station of 1st hit if it is in CSC
    else if  ( (*immrh).isCSC() ) {
      CSCDetId did(id.rawId());
      station1 = did.station();
    }
    // check next RecHit
    ConstRecHitContainer::const_iterator nexthit(ihit);
    nexthit++;
    
    if ( ( nexthit != all.end()) && (*nexthit)->isValid() ) {
      ConstMuonRecHitPointer immrh2 = dynamic_cast<const MuonTransientTrackingRecHit*>((*nexthit).get());
      DetId id2 = immrh2->geographicalId();
      
      // get station of 1st hit if it is in DT
      if ( (*immrh2).isDT()  ) {
        DTChamberId did2(id2.rawId());
        station2 = did2.station();
      }
      // otherwise get station of 1st hit if it is in CSC
      else if  ( (*immrh2).isCSC() ) {
        CSCDetId did2(id2.rawId());
        station2 = did2.station();
      }
      
      // 1st hit is in station 1 and second hit is in a different station
      // or an rpc (if station = -999 it could be an rpc hit)
      if ( (station1 != -999) && ((station2 == -999) || (station2 > station1)) ) {
	LogTrace(theCategory) << " station 1 = "<<station1 
			      <<", r = "<< (*ihit)->globalPosition().perp()
			      <<", z = "<< (*ihit)->globalPosition().z() << ", "; 
	
	LogTrace(theCategory) << " station 2 = " << station2
			      <<", r = "<<(*(nexthit))->globalPosition().perp()
			      <<", z = "<<(*(nexthit))->globalPosition().z() << ", ";
	return;
      }
    }
    else if ( (nexthit==all.end()) && (station1!=-999) ) {
      LogTrace(theCategory) << " station 1 = "<< station1
			    << ", r = " << (*ihit)->globalPosition().perp()
			    << ", z = " << (*ihit)->globalPosition().z() << ", "; 
      return;
    }
  }

  // if none of the above is satisfied, return blank vector.
  first.clear();

  return; 
}


//
// select muon hits compatible with trajectory; 
// check hits in chambers with showers
//
GlobalMuonRefitter::ConstRecHitContainer 
GlobalMuonRefitter::selectMuonHits(const Trajectory& traj, 
                                   const std::vector<int>& hits) const {

  ConstRecHitContainer muonRecHits;
  const double globalChi2Cut = 200.0;

  vector<TrajectoryMeasurement> muonMeasurements = traj.measurements(); 

  // loop through all muon hits and skip hits with bad chi2 in chambers with high occupancy      
  for (std::vector<TrajectoryMeasurement>::const_iterator im = muonMeasurements.begin(); im != muonMeasurements.end(); im++ ) {

    if ( !(*im).recHit()->isValid() ) continue;
    if ( (*im).recHit()->det()->geographicalId().det() != DetId::Muon ) {
      //      if ( ( chi2ndf < globalChi2Cut ) )
      muonRecHits.push_back((*im).recHit());
      continue;
    }  
    ConstMuonRecHitPointer immrh = dynamic_cast<const MuonTransientTrackingRecHit*>((*im).recHit().get());

    DetId id = immrh->geographicalId();
    int station = 0;
    int threshold = 0;
    double chi2Cut = 0.0;

    // get station of hit if it is in DT
    if ( (*immrh).isDT() ) {
      DTChamberId did(id.rawId());
      station = did.station();
      threshold = theHitThreshold;
      chi2Cut = theDTChi2Cut;
    }
    // get station of hit if it is in CSC
    else if ( (*immrh).isCSC() ) {
      CSCDetId did(id.rawId());
      station = did.station();
      threshold = theHitThreshold;
      chi2Cut = theCSCChi2Cut;
    }
    // get station of hit if it is in RPC
    else if ( (*immrh).isRPC() ) {
      RPCDetId rpcid(id.rawId());
      station = rpcid.station();
      threshold = theHitThreshold;
      chi2Cut = theRPCChi2Cut;
    }
    else
      continue;

    double chi2ndf = (*im).estimate()/(*im).recHit()->dimension();  

    bool keep = true;
    if ( (station>0) && (station<5) ) {
      if (hits[station-1]>threshold) keep = false;
    }   
    
    if ( (keep || (chi2ndf<chi2Cut)) && (chi2ndf<globalChi2Cut) ) {
      muonRecHits.push_back((*im).recHit());
    } else {
      LogTrace(theCategory)
	<< "Skip hit: " << id.det() << " " << station << ", " 
	<< chi2ndf << " (" << chi2Cut << " chi2 threshold) " 
	<< hits[station-1] << endl;
    }
  }
  
  // check order of rechits
  reverse(muonRecHits.begin(),muonRecHits.end());
  return muonRecHits;
}


//
// print RecHits
//
void GlobalMuonRefitter::printHits(const ConstRecHitContainer& hits) const {

  LogTrace(theCategory) << "Used RecHits: " << hits.size();
  for (ConstRecHitContainer::const_iterator ir = hits.begin(); ir != hits.end(); ir++ ) {
    if ( !(*ir)->isValid() ) {
      LogTrace(theCategory) << "invalid RecHit";
      continue; 
    }
    
    const GlobalPoint& pos = (*ir)->globalPosition();
    
    LogTrace(theCategory) 
      << "r = " << sqrt(pos.x() * pos.x() + pos.y() * pos.y())
      << "  z = " << pos.z()
      << "  dimension = " << (*ir)->dimension()
      << "  " << (*ir)->det()->geographicalId().det()
      << "  " << (*ir)->det()->subDetector();
  }

}


//
// add Trajectory* to TrackCand if not already present
//
GlobalMuonRefitter::RefitDirection
GlobalMuonRefitter::checkRecHitsOrdering(const TransientTrackingRecHit::ConstRecHitContainer& recHits) const {

  if (!recHits.empty()){
    ConstRecHitContainer::const_iterator frontHit = recHits.begin();
    ConstRecHitContainer::const_iterator backHit  = recHits.end() - 1;
    while( !(*frontHit)->isValid() && frontHit != backHit) {frontHit++;}
    while( !(*backHit)->isValid() && backHit != frontHit)  {backHit--;}

    double rFirst = (*frontHit)->globalPosition().mag();
    double rLast  = (*backHit) ->globalPosition().mag();

    if(rFirst < rLast) return insideOut;
    else if(rFirst > rLast) return outsideIn;
    else {
      LogError(theCategory) << "Impossible determine the rechits order" <<endl;
      return undetermined;
    }
  } else {
    LogError(theCategory) << "Impossible determine the rechits order" <<endl;
    return undetermined;
  }
}


//
// Convert Tracks into Trajectories with a given set of hits
//
vector<Trajectory> GlobalMuonRefitter::transform(const reco::Track& newTrack,
						 const reco::TransientTrack track,
						 TransientTrackingRecHit::ConstRecHitContainer recHitsForReFit) const {
  
  if(recHitsForReFit.size() < 2) return vector<Trajectory>();

  // Check the order of the rechits
  RefitDirection recHitsOrder = checkRecHitsOrdering(recHitsForReFit);

  // Reverse the order in the case of inconsistency between the fit direction and the rechit order
  if(theRefitDirection != recHitsOrder) reverse(recHitsForReFit.begin(),recHitsForReFit.end());

  // Fill the starting state
  TrajectoryStateOnSurface firstTSOS;
  unsigned int innerId;
  if(theRefitDirection == insideOut){
    innerId =   newTrack.innerDetId();
    firstTSOS = track.innermostMeasurementState();
  } else {
    innerId   = newTrack.outerDetId();
    firstTSOS = track.outermostMeasurementState();
  }

  if(!firstTSOS.isValid()){
    LogWarning(theCategory) << "Error wrong initial state!" << endl;
    return vector<Trajectory>();
  }

  firstTSOS.rescaleError(1000.);

  // This is the only way to get a TrajectorySeed with settable propagation direction
  PTrajectoryStateOnDet garbage1;
  edm::OwnVector<TrackingRecHit> garbage2;
  PropagationDirection propDir = 
    (firstTSOS.globalPosition().basicVector().dot(firstTSOS.globalMomentum().basicVector())>0) ? alongMomentum : oppositeToMomentum;

  if(propDir == alongMomentum && theRefitDirection == outsideIn)  propDir=oppositeToMomentum;
  if(propDir == oppositeToMomentum && theRefitDirection == insideOut) propDir=alongMomentum;
  
  TrajectorySeed seed(garbage1,garbage2,propDir);

  if(recHitsForReFit.front()->geographicalId() != DetId(innerId)){
    LogDebug(theCategory)<<"Propagation occured"<<endl;
    firstTSOS = theService->propagator(thePropagatorName)->propagate(firstTSOS, recHitsForReFit.front()->det()->surface());
    if(!firstTSOS.isValid()){
      LogDebug(theCategory)<<"Propagation error!"<<endl;
      return vector<Trajectory>();
    }
  }

/*  
  cout << " GlobalMuonRefitter : theFitter " << propDir << endl;
  cout << "                      First TSOS: " 
       << firstTSOS.globalPosition() << "  p="
       << firstTSOS.globalMomentum() << " = "
       << firstTSOS.globalMomentum().mag() << endl;
       
  cout << "                      Starting seed: "
       << " nHits= " << seed.nHits()
       << " tsos: "
       << seed.startingState().parameters().position() << "  p="
       << seed.startingState().parameters().momentum() << endl;
       
  cout << "                      RecHits: "
       << recHitsForReFit.size() << endl;
*/
       
  vector<Trajectory> trajectories = theFitter->fit(seed,recHitsForReFit,firstTSOS);
  
  if(trajectories.empty()){
    LogDebug(theCategory) << "No Track refitted!" << endl;
    return vector<Trajectory>();
  }
  
  return trajectories;
}


//
// Remove Selected Station Rec Hits
//
GlobalMuonRefitter::ConstRecHitContainer GlobalMuonRefitter::getRidOfSelectStationHits(ConstRecHitContainer hits) const
{
  ConstRecHitContainer results;
  ConstRecHitContainer::const_iterator it = hits.begin();
  for (; it!=hits.end(); it++) {

    DetId id = (*it)->geographicalId();

    //Check that this is a Muon hit that we're toying with -- else pass on this because the hacker is a moron / not careful

    if (id.det() == DetId::Tracker && theTrackerSkipSystem > 0) {
      int layer = -999;
      int disk  = -999;
      int wheel = -999;
      if ( id.subdetId() == theTrackerSkipSystem){
	//                              continue;  //caveat that just removes the whole system from refitting

	if (theTrackerSkipSystem == PXB) {
	  PXBDetId did(id.rawId());
	  layer = did.layer();
	}
	if (theTrackerSkipSystem == TIB) {
	  TIBDetId did(id.rawId());
	  layer = did.layer();
	}

	if (theTrackerSkipSystem == TOB) {
	  TOBDetId did(id.rawId());
	  layer = did.layer();
	}
	if (theTrackerSkipSystem == PXF) {
	  PXFDetId did(id.rawId());
	  disk = did.disk();
	}
	if (theTrackerSkipSystem == TID) {
	  TIDDetId did(id.rawId());
	  wheel = did.wheel();
	}
	if (theTrackerSkipSystem == TEC) {
	  TECDetId did(id.rawId());
	  wheel = did.wheel();
	}
	if (theTrackerSkipSection >= 0 && layer == theTrackerSkipSection) continue;
	if (theTrackerSkipSection >= 0 && disk == theTrackerSkipSection) continue;
	if (theTrackerSkipSection >= 0 && wheel == theTrackerSkipSection) continue;
      }
    }

    if (id.det() == DetId::Muon && theSkipStation) {
      int station = -999;
      int wheel = -999;
      if ( id.subdetId() == MuonSubdetId::DT ) {
	DTChamberId did(id.rawId());
	station = did.station();
	wheel = did.wheel();
      } else if ( id.subdetId() == MuonSubdetId::CSC ) {
	CSCDetId did(id.rawId());
	station = did.station();
      } else if ( id.subdetId() == MuonSubdetId::RPC ) {
	RPCDetId rpcid(id.rawId());
	station = rpcid.station();
      }
      if(station == theSkipStation) continue;
    }
    results.push_back(*it);
  }
  return results;
}

