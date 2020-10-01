#ifndef Geometry_TrackerNumberingBuilder_DDDCmsTrackerContruction_H
#define Geometry_TrackerNumberingBuilder_DDDCmsTrackerContruction_H

#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerStringToEnum.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>
#include <vector>
#include <memory>

class GeometricDet;
class DDCompactView;

namespace cms {
  class DDCompactView;
}

/**
 * High level class to build a tracker. It will only build subdets,
 * then call subdet builders
 */

class DDDCmsTrackerContruction {
public:
  DDDCmsTrackerContruction() = delete;
  ///takes ownership of detidShifts
  static std::unique_ptr<GeometricDet> construct(DDCompactView const& cpv, std::vector<int> const& detidShifts);
  static std::unique_ptr<GeometricDet> construct(cms::DDCompactView const& cpv, std::vector<int> const& detidShifts);
  static void printAllTrackerGeometricDetsBeforeDetIDBuilding(const GeometricDet* tracker);
};  // NB: no point having a class in which evth is static, should just use namespace...

#endif
