#ifndef DD_DDD_H_GRD
#define DD_DDD_H_GRD

// CLHEP
#include "CLHEP/Units/SystemOfUnits.h"

#include "DetectorDescription/DDBase/interface/DDException.h"

#include "DetectorDescription/DDBase/interface/DDTypes.h"
#include "DetectorDescription/DDCore/interface/DDConstant.h"
#include "DetectorDescription/DDCore/interface/DDString.h"
#include "DetectorDescription/DDCore/interface/DDVector.h"
#include "DetectorDescription/DDCore/interface/DDMap.h"
#include "DetectorDescription/DDCore/interface/DDLParserI.h"
#include "DetectorDescription/DDCore/interface/DDPosPart.h"
#include "DetectorDescription/DDCore/interface/DDName.h"
#include "DetectorDescription/DDCore/interface/DDLogicalPart.h"
#include "DetectorDescription/DDCore/interface/DDMaterial.h"
#include "DetectorDescription/DDCore/interface/DDSolid.h"


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
  typedef DDLParser Parser;
};
#endif // DD_DDD_H_GRD
