#ifndef RecoTrackerDeDx_TruncatedAverageDeDxEstimator_h
#define RecoTrackerDeDx_TruncatedAverageDeDxEstimator_h

#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include <numeric>

class TruncatedAverageDeDxEstimator: public BaseDeDxEstimator
{
public: 
 TruncatedAverageDeDxEstimator(const edm::ParameterSet& iConfig){
    m_fraction = iConfig.getParameter<double>("fraction");
 }

 virtual std::pair<float,float> dedx(const reco::DeDxHitCollection& Hits){
    int nTrunc = int( Hits.size()*m_fraction);
    double sumdedx = 0;
    for(size_t i=0;i + nTrunc <  Hits.size() ; i++){
       sumdedx+=Hits[i].charge();
    } 
   double avrdedx = (Hits.size()) ? sumdedx/(Hits.size()-nTrunc) :0.0;
   return  std::make_pair(avrdedx,-1);
 } 

private:
 float m_fraction;

};

#endif
