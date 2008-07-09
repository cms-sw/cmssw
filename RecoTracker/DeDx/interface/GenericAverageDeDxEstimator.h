#ifndef RecoTrackerDeDx_GenericAverageDeDxEstimator_h
#define RecoTrackerDeDx_GenericAverageDeDxEstimator_h

#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"

class GenericAverageDeDxEstimator: public BaseDeDxEstimator
{
public: 
 GenericAverageDeDxEstimator(float expo): m_expo(expo) {}

 virtual float dedx(const reco::DeDxHitCollection & Hits) 
 {return DeDxTools::genericAverage(Hits, m_expo); } 

private:
 float m_expo;

};

#endif
