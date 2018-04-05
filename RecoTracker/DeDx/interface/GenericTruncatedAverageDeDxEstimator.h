#ifndef RecoTrackerDeDx_GenericTruncatedAverageDeDxEstimator_h
#define RecoTrackerDeDx_GenericTruncatedAverageDeDxEstimator_h

#include "RecoTracker/DeDx/interface/BaseDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include <numeric>

class GenericTruncatedAverageDeDxEstimator: public BaseDeDxEstimator
{
public: 
 GenericTruncatedAverageDeDxEstimator(const edm::ParameterSet& iConfig){
    m_fraction = iConfig.getParameter<double>("fraction");
    m_expo = iConfig.getParameter<double>("exponent");
 }

 std::pair<float,float> dedx(const reco::DeDxHitCollection& Hits) override{
    int first = 0, last = Hits.size();
    if (m_fraction > 0) { // truncate high charge ones
       last -= int(Hits.size()*m_fraction); 
    } else {
       first += int(Hits.size()*(-m_fraction)); 
    }
    double sumdedx = 0;
    for(int i = first; i < last; i++){
       sumdedx+=pow(Hits[i].charge(),m_expo);
    } 
   double avrdedx = (last-first) ? pow(sumdedx/(last-first),1.0/m_expo) :0.0;
   return  std::make_pair(avrdedx,-1);
 } 

private:
 float m_fraction, m_expo;

};

#endif
