#ifndef Geometry_MTDNumberingBuilder_CondDBCmsMTDConstruction_H
#define Geometry_MTDNumberingBuilder_CondDBCmsMTDConstruction_H

#include "Geometry/MTDNumberingBuilder/interface/CmsMTDStringToEnum.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>
#include <memory>

class GeometricTimingDet;
class DDCompactView;
class PGeometricTimingDet;

/**
 * High level class to build a tracker. It will only build subdets,
 * then call subdet builders
 */

class CondDBCmsMTDConstruction {
public:
  CondDBCmsMTDConstruction() = delete;
  static std::unique_ptr<GeometricTimingDet> construct(const PGeometricTimingDet& pgd);
};

#endif
