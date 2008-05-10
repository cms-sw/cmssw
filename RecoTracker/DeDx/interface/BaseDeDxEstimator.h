#ifndef BaseDeDxEstimator_h
#define BaseDeDxEstimator_h
#include "DataFormats/TrackReco/interface/TrajectorySateOnDetInfo.h"

class BaseDeDxEstimator
{
public: 
 virtual float dedx(reco::TrajectorySateOnDetInfoCollection tsodis, edm::ESHandle<TrackerGeometry> tkGeom) = 0;



};

#endif
