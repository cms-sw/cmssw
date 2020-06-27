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
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class BTLRecHitsErrorEstimatorIM {
public:
  BTLRecHitsErrorEstimatorIM(const MTDGeomDet* det, const LocalPoint& lp) : det_(det), lp_(lp) {
    if (GeomDetEnumerators::isEndcap(det->type().subDetector())) {
      throw cms::Exception("BTLRecHitsErrorEstimatorIM")
          << "This is an object from Endcap. Only use it for the Barrel!" << std::endl;
    }
  }
  LocalError localError() const {
    /// position error, refer to:
    /// https://indico.cern.ch/event/825902/contributions/3455359/attachments/1858923/3054344/residual_calculation_0607.pdf
    const float positionError2 = std::pow(positionError(), 2);
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(det_->topology());
    const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());
    MeasurementPoint mp = topo.measurementPosition(lp_);
    MeasurementError simpleRect(1. / 12., 0, 1. / 12.);
    LocalError error_before = topo.localError(mp, simpleRect);
    LocalError error_modified(positionError2, error_before.xy(), error_before.yy());
    return error_modified;
  }
  static float positionError() {
    constexpr float positionError = 0.6f;
    return positionError;
  }

private:
  const MTDGeomDet* det_;
  const LocalPoint& lp_;
};

#endif
