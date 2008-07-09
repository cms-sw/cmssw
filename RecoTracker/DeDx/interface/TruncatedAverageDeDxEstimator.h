#ifndef RecoTrackerDeDx_TruncatedAverageDeDxEstimator_h
#define RecoTrackerDeDx_TruncatedAverageDeDxEstimator_h

#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include <numeric>

class TruncatedAverageDeDxEstimator: public BaseDeDxEstimator
{
public: 
 TruncatedAverageDeDxEstimator(float fraction): m_fraction(fraction) {}

 virtual float dedx(const reco::DeDxHitCollection & Hits) 
 {
  int nTrunc = int( Hits.size()*m_fraction);
  double sumdedx = 0;
  for(size_t i=0;i + nTrunc <  Hits.size() ; i++)
   {
     sumdedx+=Hits[i].charge();
   } 
 double avrdedx = (Hits.size()) ? sumdedx/(Hits.size()-nTrunc) :0.0;

  return  avrdedx;
 } 

private:
 float m_fraction;

};

#endif
