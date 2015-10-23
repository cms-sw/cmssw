#ifndef FASTSIMULATION_TRACKING_FASTTRACKERRECHITSPLITTER_H
#define FASTSIMULATION_TRACKING_FASTTRACKERRECHITSPLITTER_H

#include <vector>
#include "FastSimulation/Tracking/interface/TrajectorySeedHitCandidate.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastSingleTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastMatchedTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastProjectedTrackerRecHit.h"

class FastTrackerRecHitSplitter {
    
    public:
    
    FastTrackerRecHitSplitter(){;}
    ~FastTrackerRecHitSplitter(){;}
    
    inline void split(const FastTrackerRecHit & hitIn,edm::OwnVector<TrackingRecHit> & hitsOut) const {
	
	if(hitIn.dimension()==1 || hitIn.isPixel())
	    hitsOut.push_back( hitIn.clone() );
	
	else if(hitIn.isProjected())
	    hitsOut.push_back(buildSplitStripHit(static_cast<const FastProjectedTrackerRecHit &>(hitIn).originalHit()));
	
	else if(hitIn.isMatched()){
	    hitsOut.push_back(buildSplitStripHit(static_cast<const FastMatchedTrackerRecHit &>(hitIn).firstHit()));
	    hitsOut.push_back(buildSplitStripHit(static_cast<const FastMatchedTrackerRecHit &>(hitIn).secondHit()));
	}
	
	else{
	    hitsOut.push_back(buildSplitStripHit(static_cast<const FastSingleTrackerRecHit &>(hitIn)));
	}
    }

    
    private:
    
    inline FastSingleTrackerRecHit * buildSplitStripHit (const FastSingleTrackerRecHit & hit) const {
	FastSingleTrackerRecHit * newHit = hit.clone();
	newHit->set2D(newHit->detUnit()->type().isEndcap());
	return newHit;
    }

};

#endif
