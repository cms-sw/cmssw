#ifndef FASTSIMULATION_TRACKING_HITMASKHELPER_H
#define FASTSIMULATION_TRACKING_HITMASKHELPER_H

#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"
#include "vector"

class HitMaskHelper {
    
    public:

    HitMaskHelper(const std::vector<bool> * hitMasks) : hitMasks_(hitMasks) { ; }

    bool mask(const FastTrackerRecHit * hit) const{
	for(unsigned int i = 0;i<hit->nIds();i++){
	    if(!(*hitMasks_)[hit->id(i)]){
		return false;
	    }
	}
	return true;
    }

    private:
    
    HitMaskHelper();
    const std::vector<bool> * hitMasks_;
};

#endif
