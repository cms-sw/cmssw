#ifndef FASTSIMULATION_TRACKING_FASTTRACKINGHELPER_H
#define FASTSIMULATION_TRACKING_FASTTRACKINGHELPER_H

#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"

namespace fastTrackingHelper {
    
    template<class T> inline void setRecHitCombinationIndex(edm::OwnVector<T> & recHits,int32_t icomb){
	for(auto & recHit : recHits){
	    if(!trackerHitRTTI::isFast(recHit)){
		throw cms::Exception("fastTrackingHelpers::setRecHitCombinationIndex: one of hits in OwnVector is non-fastsim");
	    }
	    static_cast<FastTrackerRecHit &>(recHit).setRecHitCombinationIndex(icomb);
	}
    }

    // get recHitCombination for TrackCandidate and TrajectorySeed
    template<class T> int32_t getRecHitCombinationIndex(const T & object){
	// seed must have at least one hit
	if(object.recHits().first == object.recHits().second){
	    throw cms::Exception("fastTrackingHelpers::getRecHitCombinationIndex") << "  given object has 0 hits" << std::endl;
	}

	const TrackingRecHit & recHit = *object.recHits().first;
	if(!trackerHitRTTI::isFast(recHit)){
	    throw cms::Exception("fastTrackingHelpers::setRecHitCombinationIndex") << "  one of hits in OwnVector is non-fastsim" << std::endl;
	}
	// cast and return combination index
	return static_cast<const FastTrackerRecHit &>(recHit).recHitCombinationIndex();
	// return combination index of first hit
    }
    
}

#endif
