#ifndef TruncatedAverageDeDxEstimator_h
#define TruncatedAverageDeDxEstimator_h

#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "DataFormats/TrackReco/interface/TrajectorySateOnDetInfo.h"
#include <numeric>

class TruncatedAverageDeDxEstimator: public BaseDeDxEstimator
{
public: 
 TruncatedAverageDeDxEstimator(float fraction): m_fraction(fraction) {}

 virtual float dedx(reco::TrajectorySateOnDetInfoCollection tsodis, edm::ESHandle<TrackerGeometry> tkGeom){
    if(tsodis.size()<=0) return 0;

    std::vector<double> ChargeN;
    for(unsigned int i=0;i<tsodis.size();i++){
       ChargeN.push_back(tsodis[i].chargeOverPath(tkGeom));
    }
    std::sort(ChargeN.begin(), ChargeN.end() );

    int     nTrunc = int( ChargeN.size()*m_fraction);
    double sumdedx = 0;
    for(unsigned int i=0;i + nTrunc <  ChargeN.size() ; i++){
       sumdedx+=ChargeN[i];
    } 
    return  sumdedx/(ChargeN.size()-nTrunc);
 } 

private:
 float m_fraction;

};

#endif
