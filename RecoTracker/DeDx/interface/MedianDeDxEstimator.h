#ifndef RecoTrackerDeDx_MedianDeDxEstimator_h
#define RecoTrackerDeDx_MedianDeDxEstimator_h

#include "RecoTracker/DeDx/interface/BaseDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"

class MedianDeDxEstimator: public BaseDeDxEstimator
{
public: 
 MedianDeDxEstimator(const edm::ParameterSet& iConfig){
 }

 std::pair<float,float> dedx(const reco::DeDxHitCollection & Hits) override{
    if(Hits.empty())return std::make_pair(-1,-1);
    return std::make_pair(Hits[Hits.size()/2].charge(),-1); 
 } 
};

#endif
