#ifndef RecoLocalFastTime_FTLClusterizer_BTLRecHitsErrorEstimatorIM_H
#define RecoLocalFastTime_FTLClusterizer_BTLRecHitsErrorEstimatorIM_H 1

//-----------------------------------------------------------------------------
// \class         BTLRecHitsErrorEstimatorIM
// Used to improve the local error of recHits and TrackingrecHits in BTL
//-----------------------------------------------------------------------------

#include "Geometry/MTDGeometryBuilder/interface/MTDGeomDetUnit.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"

#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"

#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/Utilities/interface/Exception.h"

class BTLRecHitsErrorEstimatorIM {
public:
  BTLRecHitsErrorEstimatorIM(const MTDGeomDet* det, const LocalPoint& lp) : det_(det), lp_(lp) {}
  LocalError localError() const {
    if (GeomDetEnumerators::isEndcap(det_->type().subDetector())) {
      throw cms::Exception("BTLRecHitsErrorEstimatorIM") << "This is an object from Endcap. Only use it for the Barrel!" << std::endl;
      return LocalError(0, 0, 0);
    }
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(det_->topology());
    const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());
    MeasurementPoint mp = topo.measurementPosition(lp_);
    MeasurementError simpleRect(1. / 12., 0, 1. / 12.);
    LocalError error_before = topo.localError(mp, simpleRect);
    LocalError error_modified(error_before.xx(), error_before.xy(), 0.36);
    return error_modified;
  }

private:
  const MTDGeomDet* det_;
  const LocalPoint& lp_;
};

#endif
