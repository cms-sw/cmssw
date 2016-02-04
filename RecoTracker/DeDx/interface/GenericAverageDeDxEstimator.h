#ifndef RecoTrackerDeDx_GenericAverageDeDxEstimator_h
#define RecoTrackerDeDx_GenericAverageDeDxEstimator_h

#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"

class GenericAverageDeDxEstimator: public BaseDeDxEstimator
{
public: 
 GenericAverageDeDxEstimator(float expo): m_expo(expo) {}

 virtual std::pair<float,float> dedx(const reco::DeDxHitCollection& Hits) 
 {return std::make_pair(DeDxTools::genericAverage(Hits, m_expo),-1); } 

private:
 float m_expo;

};

#endif
