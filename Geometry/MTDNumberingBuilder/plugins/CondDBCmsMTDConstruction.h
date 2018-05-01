#ifndef Geometry_MTDNumberingBuilder_CondDBCmsMTDConstruction_H
#define Geometry_MTDNumberingBuilder_CondDBCmsMTDConstruction_H

#include "Geometry/MTDNumberingBuilder/interface/CmsMTDStringToEnum.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

class GeometricTimingDet;
class DDCompactView;
class PGeometricTimingDet;

/**
 * High level class to build a tracker. It will only build subdets,
 * then call subdet builders
 */

class CondDBCmsMTDConstruction {
 public:
  CondDBCmsMTDConstruction();
  const GeometricTimingDet* construct( const PGeometricTimingDet& pgd );
  
 protected:
  
};

#endif
