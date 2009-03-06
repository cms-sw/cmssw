#ifndef TruncatedAverageDeDxEstimator_h
#define TruncatedAverageDeDxEstimator_h

#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include <numeric>

class TruncatedAverageDeDxEstimator: public BaseDeDxEstimator
{
public: 
 TruncatedAverageDeDxEstimator(float fraction): m_fraction(fraction) {}

 virtual float dedx(const reco::TrackDeDxHits & trackWithHits) 
 {
  int nTrunc = int( trackWithHits.second.size()*m_fraction);
  double sumdedx = 0;
  for(size_t i=0;i + nTrunc <  trackWithHits.second.size() ; i++)
   {
     sumdedx+=trackWithHits.second[i].charge();
   } 
 double avrdedx = (trackWithHits.second.size()) ? sumdedx/(trackWithHits.second.size()-nTrunc) :0.0;

  return  avrdedx;
 } 

private:
 float m_fraction;

};

#endif
