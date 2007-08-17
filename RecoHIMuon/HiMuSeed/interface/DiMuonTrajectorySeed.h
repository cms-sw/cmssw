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
  DiMuonTrajectorySeed( 
                        TrajectoryStateOnSurface tsos, 
                        const FreeTrajectoryState& ftsmuon, 
			const TrackingRecHit*  rh, 
			int aMult,
			int det 
		      );

// access

  TrajectorySeed TrajSeed(){return TrajectorySeed(startingState(),hits(),direction());};

  recHitContainer hits(){ return theRecHits; };

  FreeTrajectoryState getMuon() {return theFtsMuon;};
  
  int getMult(){return theLowMult;};
  
  range recHits() const{std::cout<<" Number of RecHits "<<theRecHits.size(); return std::make_pair(theRecHits.begin(), theRecHits.end());};
  
  PropagationDirection direction() const{return thePropagationDirection;}
  
  PTrajectoryStateOnDet startingState(){return *PTraj;}
      
 private:
 TransientTrackingRecHit::ConstRecHitPointer rh;
 recHitContainer theRecHits;
 FreeTrajectoryState theFtsMuon;
 PropagationDirection thePropagationDirection;
 boost::shared_ptr<PTrajectoryStateOnDet> PTraj;
 int                 theLowMult;
 int                 theDetId;
 TrajectoryStateTransform transformer;
};

#endif // Tracker_TrajectorySeed_H











