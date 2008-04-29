/**
 *  Class: GlobalMuonRefitter
 *
 *  Description:
 *
 *
 *  $Date: 2008/02/25 22:17:48 $
 *  $Revision: 1.1 $
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

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "Utilities/Timing/interface/TimerStack.h"

#include "RecoTracker/TkTrackingRegions/interface/TkTrackingRegionsMargin.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"

using namespace std;
using namespace edm;

//----------------
// Constructors --
//----------------

GlobalMuonRefitter::GlobalMuonRefitter(const edm::ParameterSet& par,
							 const MuonServiceProxy* service) : 
  TrackTransformer(par),
  theService(service) {

  theCategory = par.getUntrackedParameter<string>("Category", "Muon|RecoMuon|GlobalMuon|GlobalMuonRefitter");

  theLayerMeasurements = new MuonDetLayerMeasurements(par.getParameter<InputTag>("DTRecSegmentLabel"),
						      par.getParameter<InputTag>("CSCRecSegmentLabel"),
						      par.getParameter<InputTag>("RPCRecSegmentLabel"));
  
  theHitThreshold = par.getParameter<int>("HitThreshold");
  theDTChi2Cut  = par.getParameter<double>("Chi2CutDT");
  theCSCChi2Cut = par.getParameter<double>("Chi2CutCSC");
  theRPCChi2Cut = par.getParameter<double>("Chi2CutRPC");

}

//--------------
// Destructor --
//--------------

GlobalMuonRefitter::~GlobalMuonRefitter() {

  if (theLayerMeasurements) delete theLayerMeasurements;
  
}

//
// set Event
//
void GlobalMuonRefitter::setEvent(const edm::Event& event) {
  
  theEvent = &event;
  theLayerMeasurements->setEvent(event);  
  setServices(theService->eventSetup());
}

//
// build a combined tracker-muon trajectory
//
Trajectory GlobalMuonRefitter::refit(const reco::Track& globalTrack, const int theMuonHitsOption) const {

  // MuonHitsOption: 0 - tracker only
  //                 1 - include all muon hits
  //                 2 - include only first muon hit(s)
  //                 3 - include only selected muon hits

  vector<int> stationHits(4,0);

  ConstRecHitContainer allRecHits; // all muon rechits
  ConstRecHitContainer fmsRecHits; // only first muon rechits
  ConstRecHitContainer selectedRecHits; // selected muon rechits

  LogTrace(theCategory) << " *** GlobalMuonRefitter *** " << endl;

  LogTrace(theCategory) << " Track momentum before refit: " << globalTrack.pt() << endl;

  reco::TransientTrack track(globalTrack,magneticField(),trackingGeometry());

  allRecHits = getTransientRecHits(track);
  
  LogTrace(theCategory) << " Hits size: " << allRecHits.size() << endl;

  // check and select muon measurements and
  // measure occupancy in muon stations
  checkMuonHits(globalTrack, allRecHits, fmsRecHits, stationHits);
 
  // full track with all muon hits
  vector <Trajectory> globalTraj = transform(globalTrack, track, allRecHits);

//  LogTrace(theCategory) << " Initial pt: " << globalTraj.front().firstMeasurement().updatedState().globalParameters().momentum().perp();
  LogTrace(theCategory) << " Initial trajectory state: " << globalTraj.front().lastMeasurement().updatedState().freeState()->parameters() << endl;
    
//    printHits(allRecHits);

  vector <Trajectory> outputTraj;

  if (theMuonHitsOption == 1 ) {
    outputTraj.push_back(globalTraj.front());
    // for testing only - the result should be the same as before the refit
  }     

  if (theMuonHitsOption == 2 ) {
    outputTraj = transform(globalTrack, track, fmsRecHits);
  }     

  if (theMuonHitsOption == 3 ) {
    selectedRecHits = selectMuonHits(globalTraj.front(),stationHits);
    LogTrace(theCategory) << " Selected hits size: " << selectedRecHits.size() << endl;  
    outputTraj = transform(globalTrack, track, selectedRecHits);
  }     
  
  if (outputTraj.size()) {
    LogTrace(theCategory) << "Refitted pt: " << outputTraj.front().firstMeasurement().updatedState().globalParameters().momentum().perp();
    return outputTraj.front();
  }    
}

//
//
//
void GlobalMuonRefitter::checkMuonHits(const reco::Track& muon, 
				       ConstRecHitContainer& all,
				       ConstRecHitContainer& first,
				       std::vector<int>& hits) const {

  LogTrace(theCategory) << " GlobalMuonRefitter::checkMuonHits " << endl;

  int dethits[4];
  for ( int i=0; i<4; i++ ) hits[i]=dethits[i]=0;

//  MuonTransientTrackingRecHitBuilder muonRecHitBuilder(theService->trackingGeometry());
  
//  ConstRecHitContainer muonRecHits = muonRecHitBuilder.build(muon.recHitsBegin(),muon.recHitsEnd());
    
  // loop through all muon hits and calculate the maximum # of hits in each chamber
  for (ConstRecHitContainer::const_iterator imrh = all.begin(); imrh != all.end(); imrh++ ) {
//  for (trackingRecHit_iterator imrh = muon.recHitsBegin(); imrh != muon.recHitsEnd(); imrh++ ) {
//  for (ConstRecHitContainer::const_iterator imrh = muonRecHits.begin(); imrh != muonRecHits.end(); imrh++ ) {
        
    if ( (*imrh != 0 ) && !(*imrh)->isValid() ) continue;
  
    int station = 0;
    int detRecHits = 0;
      
    DetId id = (*imrh)->geographicalId();

    // Skip tracker hits
    if (id.det()!=DetId::Muon) continue;
      
    const DetLayer* layer = theService->detLayerGeometry()->idToLayer(id);
    
    MuonRecHitContainer dRecHits = theLayerMeasurements->recHits(layer);
      
      // get station of hit if it is in DT
      if ( id.subdetId() == MuonSubdetId::DT ) {
        DTChamberId did(id.rawId());
        station = did.station();
        float coneSize = 10.0;
	
        bool hitUnique = false;
        ConstRecHitContainer all2dRecHits;
        for (MuonRecHitContainer::const_iterator ir = dRecHits.begin(); ir != dRecHits.end(); ir++ ) {
          if ( (**ir).dimension() == 2 ) {
            hitUnique = true;
            if ( all2dRecHits.size() > 0 ) {
              for (ConstRecHitContainer::const_iterator iir = all2dRecHits.begin(); iir != all2dRecHits.end(); iir++ ) 
		if (((*iir)->localPosition()-(*ir)->localPosition()).mag()<0.01) hitUnique = false;
            } //end of if ( all2dRecHits.size() > 0 )
            if ( hitUnique ) all2dRecHits.push_back((*ir).get()); //FIXME!!
          } else {
            ConstRecHitContainer sRecHits = (**ir).transientHits();
            for (ConstRecHitContainer::const_iterator iir = sRecHits.begin(); iir != sRecHits.end(); iir++ ) {
              if ( (*iir)->dimension() == 2 ) {
                hitUnique = true;
                if ( !all2dRecHits.empty() ) {
                  for (ConstRecHitContainer::const_iterator iiir = all2dRecHits.begin(); iiir != all2dRecHits.end(); iiir++ ) 
		    if (((*iiir)->localPosition()-(*iir)->localPosition()).mag()<0.01) hitUnique = false;
                }//end of if ( all2dRecHits.size() > 0 )
              }//end of if ( (*iir).dimension() == 2 ) 
              if ( hitUnique )
		all2dRecHits.push_back(*iir);
              break;
            }//end of for sRecHits
          }// end of else
	}// end of for loop over dRecHits
	for (ConstRecHitContainer::const_iterator ir = all2dRecHits.begin(); ir != all2dRecHits.end(); ir++ ) {
	  double rhitDistance = ((*ir)->localPosition()-(**imrh).localPosition()).mag();
	  if ( rhitDistance < coneSize ) detRecHits++;
//	  LogTrace(theCategory) << " Station " << station << " DT "<<(*ir)->dimension()<<" " << (*ir)->localPosition()
//						      << " Distance: "<< rhitDistance<<" recHits: "<< detRecHits;
	}// end of for all2dRecHits
      }// end of if DT
      // get station of hit if it is in CSC
      else if ( id.subdetId() == MuonSubdetId::CSC ) {
	CSCDetId did(id.rawId());
	station = did.station();
	
	float coneSize = 10.0;
	
	for (MuonRecHitContainer::const_iterator ir = dRecHits.begin(); ir != dRecHits.end(); ir++ ) {
	  double rhitDistance = ((**ir).localPosition()-(**imrh).localPosition()).mag();
	  if ( rhitDistance < coneSize ) detRecHits++;
//	  LogTrace(theCategory) << " Station " << station << " CSC "<<(**ir).dimension()<<" "<<(**ir).localPosition()
//                                                  << " Distance: "<< rhitDistance<<" recHits: "<<detRecHits;
	}
      }
      // get station of hit if it is in RPC
      else if ( id.subdetId() == MuonSubdetId::RPC ) {
	RPCDetId rpcid(id.rawId());
	station = rpcid.station();
	float coneSize = 100.0;
	for (MuonRecHitContainer::const_iterator ir = dRecHits.begin(); ir != dRecHits.end(); ir++ ) {
	  double rhitDistance = ((**ir).localPosition()-(**imrh).localPosition()).mag();
	  if ( rhitDistance < coneSize ) detRecHits++;
//	  LogTrace(theCategory)<<" Station "<<station<<" RPC "<<(**ir).dimension()<<" "<< (**ir).localPosition()
//						     <<" Distance: "<<rhitDistance<<" recHits: "<<detRecHits;
	}
      }
      else {
        LogError(theCategory)<<" Wrong Hit Type ";
	continue;      
      }
      
      if ( (station > 0) && (station < 5) ) {
	int detHits = dRecHits.size();
	dethits[station-1] += detHits;
	if ( detRecHits > hits[station-1] ) hits[station-1] = detRecHits;
      }

//    all.push_back((*imrh).get()); //FIXME: may need fast assignment on above

  } // end of loop over muon rechits

  for ( int i = 0; i < 4; i++ ) {
      LogTrace(theCategory) <<"Station "<<i+1<<": "<<hits[i]<<" "<<dethits[i] <<endl; 
  }     
  
  //
  // check order of muon measurements
  //
  LogTrace(theCategory) << "CheckMuonHits: "<<all.size();

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
    else if ( (nexthit == all.end()) && (station1 != -999) ) {
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
    else {
      continue;
    }

    double chi2ndf = (*im).estimate()/(*im).recHit()->dimension();  

    bool keep = true;
    if ( (station > 0) && (station < 5) ) {
      if ( hits[station-1] > threshold ) keep = false;
    }   
    
    if ( (keep || ( chi2ndf < chi2Cut )) && ( chi2ndf < globalChi2Cut ) ) {
      muonRecHits.push_back((*im).recHit());
    } else {
      LogTrace(theCategory)
	<< "Skip hit: " << id.det() << " " << station << ", " 
	<< chi2ndf << " (" << chi2Cut << " chi2 threshold) " 
	<< hits[station-1] << endl;
    }

  }
  
  //
  // check order of rechits
  //
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

/*
//
// add Trajectory* to TrackCand if not already present
//
*/

GlobalMuonRefitter::RefitDirection
GlobalMuonRefitter::checkRecHitsOrdering(const TransientTrackingRecHit::ConstRecHitContainer& recHits) const {

  if (!recHits.empty()){
    ConstRecHitContainer::const_iterator frontHit = recHits.begin();
    ConstRecHitContainer::const_iterator backHit  = recHits.end() - 1;
    while( !(*frontHit)->isValid() && frontHit != backHit) {frontHit++;}
    while( !(*backHit)->isValid() && backHit != frontHit)  {backHit--;}

    double rFirst = (*frontHit)->globalPosition().mag();
    double rLast  = (*backHit) ->globalPosition().mag();

    if(rFirst < rLast) return inToOut;
    else if(rFirst > rLast) return outToIn;
    else{
      LogError(theCategory) << "Impossible determine the rechits order" <<endl;
      return undetermined;
    }
  }
  else{
    LogError(theCategory) << "Impossible determine the rechits order" <<endl;
    return undetermined;
  }
}

