#ifndef RecoTrackerMeasurementDetClusterFilterPayload_H
#define	RecoTrackerMeasurementDetClusterFilterPayload_H

#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"

class SiSitripCluster;
struct ClusterFilterPayload final : public MeasurementEstimator::OpaquePayload {
  ~ClusterFilterPayload(){}

  ClusterFilterPayload(unsigned int id, SiSitripCluster const	* mono, SiSitripCluster const	* stereo=nullptr) : detId(id), cluster{mono,stereo}{}
  unsigned int detId=0;
  SiSitripCluster const * cluster[2] = {nullptr,nullptr};

};



#endif
