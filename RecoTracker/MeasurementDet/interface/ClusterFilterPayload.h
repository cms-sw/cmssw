#ifndef RecoTrackerMeasurementDetClusterFilterPayload_H
#define	RecoTrackerMeasurementDetClusterFilterPayload_H

#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"

class SiStripCluster;
struct ClusterFilterPayload final : public MeasurementEstimator::OpaquePayload {
  ~ClusterFilterPayload() override{}

  ClusterFilterPayload(unsigned int id, SiStripCluster const	* mono, SiStripCluster const	* stereo=nullptr) : detId(id), cluster{mono,stereo}{ tag=myTag;}
  unsigned int detId=0;
  SiStripCluster const * cluster[2] = {nullptr,nullptr};

  static constexpr int myTag = 123;
};



#endif
