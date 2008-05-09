#ifndef Tracker_DiMuonTrajectorySeed_H
#define Tracker_DiMuonTrajectorySeed_H
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
//#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <boost/shared_ptr.hpp>
#include <vector>

class DetLayer;
class DiMuonTrajectorySeed : public TrajectorySeed {

public:

typedef edm::OwnVector<TrackingRecHit> recHitContainer;

// construct
  DiMuonTrajectorySeed( const TrajectoryMeasurement& mtm0, const FreeTrajectoryState& ftsmuon, int aMult=1 ): 
                        theFtsMuon(ftsmuon),
		        thePropagationDirection(oppositeToMomentum),
		        theLowMult(aMult)
		                                                   {
								   theTrajMeasurements.push_back(mtm0);
                                                                   //const TrackingRecHit* outerHit = mtm0.recHit()->hit();   
								   //theRecHits.push_back(outerHit->clone());
                                                                   //theRecHits.push_back(mtm0.recHit());` 
								   } 

// access

  TrajectorySeed TrajSeed(){return TrajectorySeed(startingState(),hits(),direction());};

  recHitContainer hits(){ return theRecHits; };

  FreeTrajectoryState getMuon() {return theFtsMuon;};
  
  int getMult(){return theLowMult;};
  
  range recHits() const{std::cout<<" Number of RecHits "<<theRecHits.size(); return std::make_pair(theRecHits.begin(), theRecHits.end());};
  
  PropagationDirection direction() const{return thePropagationDirection;}
      
 private:
 std::vector<TrajectoryMeasurement> theTrajMeasurements;
 edm::OwnVector<TrackingRecHit>     theRecHits;
 FreeTrajectoryState                theFtsMuon;
 PropagationDirection               thePropagationDirection;
 int                                theLowMult;
};

#endif // Tracker_TrajectorySeed_H











