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

namespace DDDCmsTrackerContruction {
  std::unique_ptr<GeometricDet> construct(DDCompactView const& cpv, std::vector<int> const& detidShifts);
  std::unique_ptr<GeometricDet> construct(cms::DDCompactView const& cpv, std::vector<int> const& detidShifts);
  void printAllTrackerGeometricDetsBeforeDetIDBuilding(const GeometricDet* tracker);
};  // namespace DDDCmsTrackerContruction

#endif
