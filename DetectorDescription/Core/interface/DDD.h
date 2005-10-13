#ifndef DD_DDD_H_GRD
#define DD_DDD_H_GRD

// CLHEP
#include "CLHEP/Units/SystemOfUnits.h"

#include "DetectorDescription/Base/interface/DDException.h"

#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDConstant.h"
#include "DetectorDescription/Core/interface/DDString.h"
#include "DetectorDescription/Core/interface/DDVector.h"
#include "DetectorDescription/Core/interface/DDMap.h"
#include "DetectorDescription/Core/interface/DDLParserI.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"


namespace DD {
  // base types
  typedef DDName Name;
  typedef DDNumeric Numeric;
  typedef DDConstant Constant;
  typedef DDVector Vector;
  typedef DDString String;
  typedef DDMap Map;
  typedef DDException Exception;
  typedef DDTranslation Translation;
  typedef DDRotation Rotation;
  typedef DDRotationMatrix RotationMatrix;

  // components
  typedef DDLogicalPart LogicalPart;
  typedef DDMaterial Material;
  typedef DDSolid Solid;
  typedef DDSolidFactory SolidFactory;

  // the parser
  //  typedef DDLParser Parser;
};
#endif // DD_DDD_H_GRD
