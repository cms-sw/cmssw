#ifndef Geometry_TrackerNumberingBuilder_CondDBCmsTrackerConstruction_H
#define Geometry_TrackerNumberingBuilder_CondDBCmsTrackerConstruction_H

#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerStringToEnum.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

class GeometricDet;
class DDCompactView;
class PGeometricDet;

/**
 * High level class to build a tracker. It will only build subdets,
 * then call subdet builders
 */

class CondDBCmsTrackerConstruction {
 public:
  CondDBCmsTrackerConstruction() = delete;
  static std::unique_ptr<GeometricDet> construct( const PGeometricDet& pgd );
};

#endif
