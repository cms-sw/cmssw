#ifndef RecoTrackerDeDx_GenericAverageDeDxEstimator_h
#define RecoTrackerDeDx_GenericAverageDeDxEstimator_h

#include "RecoTracker/DeDx/interface/BaseDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"

class GenericAverageDeDxEstimator: public BaseDeDxEstimator
{
public: 
 GenericAverageDeDxEstimator(const edm::ParameterSet& iConfig){
    m_expo = iConfig.getParameter<double>("exponent");
 }

 std::pair<float,float> dedx(const reco::DeDxHitCollection& Hits) override {
    float result=0;
    size_t n = Hits.size();
    for(size_t i = 0; i< n; i ++){
       result+=pow(Hits[i].charge(),m_expo); 
    }
    return std::make_pair( ((n>0)?pow(result/n,1./m_expo):0.0) ,-1); 
 } 

private:
 float m_expo;

};

#endif
