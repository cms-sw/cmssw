#ifndef Tracker_DiMuonTrajectorySeed_H
#define Tracker_DiMuonTrajectorySeed_H
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include <vector>

class DiMuonTrajectorySeed : public TrajectorySeed {

public:

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

  std::vector<TrajectoryMeasurement> measurements() const {return theTrajMeasurements;};
  
  FreeTrajectoryState getMuon() {return theFtsMuon;};
  
  int getMult(){return theLowMult;};
  
  range recHits() const{return std::make_pair(theRecHits.begin(), theRecHits.end());};
  
  PropagationDirection direction() const{return thePropagationDirection;}
      
 private:
 std::vector<TrajectoryMeasurement> theTrajMeasurements;
 edm::OwnVector<TrackingRecHit>     theRecHits;
 FreeTrajectoryState                theFtsMuon;
 PropagationDirection               thePropagationDirection;
 int                                theLowMult;
};

#endif // Tracker_TrajectorySeed_H











