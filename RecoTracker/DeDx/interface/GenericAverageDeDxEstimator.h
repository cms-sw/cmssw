#ifndef BaseDeDxEstimator_h
#define baseDeDxEstimator_h

#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"

class GenericAverageDeDxEstimator: public BaseDeDxEstimator
{
public: 
 GenericAverageDeDxEstimator(float expo): m_expo(expo) {}

 virtual float dedx(const reco::TrackDeDxHits & trackWithHits) 
 {return DeDxTools::genericAverage(trackWithHits.second,m_expo); } 

private:
 float m_expo;

};

#endif
