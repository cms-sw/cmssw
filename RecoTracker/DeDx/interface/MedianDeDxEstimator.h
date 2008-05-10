#ifndef MedianDeDxEstimator_h
#define MedianDeDxEstimator_h

#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "DataFormats/TrackReco/interface/TrajectorySateOnDetInfo.h"

class MedianDeDxEstimator: public BaseDeDxEstimator
{
public: 
 MedianDeDxEstimator(float expo) {}
 virtual float dedx(reco::TrajectorySateOnDetInfoCollection tsodis, edm::ESHandle<TrackerGeometry> tkGeom){
    if(tsodis.size()<=0)return 0;

    std::vector<double> ChargeN;
    for(unsigned int i=0;i<tsodis.size();i++){
       ChargeN.push_back(tsodis[i].chargeOverPath(tkGeom));
    }
    std::sort(ChargeN.begin(), ChargeN.end() );

    return  ChargeN[ChargeN.size()/2]; 
 } 
};

#endif
