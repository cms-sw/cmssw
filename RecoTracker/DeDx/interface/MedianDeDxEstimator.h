#ifndef MedianDeDxEstimator_h
#define MedianDeDxEstimator_h

#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"

class MedianDeDxEstimator: public BaseDeDxEstimator
{
public: 
 MedianDeDxEstimator(float expo) {}

 virtual float dedx(const reco::TrackDeDxHits & trackWithHits) 
 {return trackWithHits.second[trackWithHits.second.size()/2].charge(); } 
};

#endif
