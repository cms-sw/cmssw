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

 virtual std::pair<float,float> dedx(const reco::DeDxHitCollection& Hits){
    int nTrunc = int( Hits.size()*m_fraction);
    double sumdedx = 0;
    for(size_t i=0;i + nTrunc <  Hits.size() ; i++){
       sumdedx+=pow(Hits[i].charge(),m_expo);
    } 
   double avrdedx = (Hits.size()) ? pow(sumdedx/(Hits.size()-nTrunc),1.0/m_expo) :0.0;
   return  std::make_pair(avrdedx,-1);
 } 

private:
 float m_fraction, m_expo;

};

#endif
