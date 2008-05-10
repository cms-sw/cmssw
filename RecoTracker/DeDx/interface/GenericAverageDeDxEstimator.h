#ifndef GenericAverageDeDxEstimator_h
#define GenericAverageDeDxEstimator_h

#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "DataFormats/TrackReco/interface/TrajectorySateOnDetInfo.h"

class GenericAverageDeDxEstimator: public BaseDeDxEstimator
{
public: 
 GenericAverageDeDxEstimator(float expo): m_expo(expo) {}

 virtual float dedx(reco::TrajectorySateOnDetInfoCollection tsodis, edm::ESHandle<TrackerGeometry> tkGeom){ 
    double result=0;
    size_t n = tsodis.size();
    if(n<=0) return 0;

    for(size_t i = 0; i< n; i ++){
       result += pow(tsodis[i].chargeOverPath(tkGeom),m_expo);
    }
    return pow(result/n,1./m_expo);
 } 

private:
 float m_expo;

};

#endif
