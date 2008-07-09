#ifndef RecoTrackerDeDx_MedianDeDxEstimator_h
#define RecoTrackerDeDx_MedianDeDxEstimator_h

#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"

class MedianDeDxEstimator: public BaseDeDxEstimator
{
public: 
 MedianDeDxEstimator(float expo) {}

 virtual float dedx(const reco::DeDxHitCollection & Hits) 
 {return Hits[Hits.size()/2].charge(); } 
};

#endif
