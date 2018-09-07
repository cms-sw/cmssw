#ifndef Geometry_MTDNumberingBuilder_DDDCmsMTDConstruction_H
#define Geometry_MTDNumberingBuilder_DDDCmsMTDConstruction_H

#include "Geometry/MTDNumberingBuilder/interface/CmsMTDStringToEnum.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>
#include <vector>

class GeometricTimingDet;
class DDCompactView;

/**
 * High level class to build a tracker. It will only build subdets,
 * then call subdet builders
 */

class DDDCmsMTDConstruction
{
public:
  DDDCmsMTDConstruction( void );
  const GeometricTimingDet* construct( const DDCompactView* cpv, std::vector<int> detidShifts);
  
protected:

  std::string attribute;  
  CmsMTDStringToEnum theCmsMTDStringToEnum;
};

#endif
