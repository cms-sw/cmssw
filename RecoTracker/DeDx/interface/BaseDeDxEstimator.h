#ifndef BaseDeDxEstimator_h
#define BaseDeDxEstimator_h
#include "DataFormats/TrackReco/interface/TrajectorySateOnDetInfo.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

class BaseDeDxEstimator
{
public: 
 virtual Measurement1D  dedx(reco::TrajectorySateOnDetInfoCollection tsodis, edm::ESHandle<TrackerGeometry> tkGeom) = 0;



};

#endif
