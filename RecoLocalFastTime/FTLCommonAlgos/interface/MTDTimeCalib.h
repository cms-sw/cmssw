#ifndef RecoLocalFastTime_FTLCommonAlgos_MTDTimeCalib_H
#define RecoLocalFastTime_FTLCommonAlgos_MTDTimeCalib_H 1

#include "DataFormats/ForwardDetId/interface/MTDDetId.h"

#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MTDTimeCalib {
public:
  //constructor & destructor
  MTDTimeCalib(edm::ParameterSet const& conf, const MTDGeometry* geom, const MTDTopology* topo);
  ~MTDTimeCalib(){};

  //accessors
  float getTimeCalib(const MTDDetId& id) const;

private:
  const MTDGeometry* geom_;
  const MTDTopology* topo_;
  float btlTimeOffset_;
  float etlTimeOffset_;

  //specific paramters from BTL simulation
  float btlLightCollTime_;
  float btlLightCollSlope_;
};

#endif
