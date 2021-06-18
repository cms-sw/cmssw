#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLAlgorithm.h"
#include "DetectorDescription/Parser/src/DDLAssembly.h"
#include "DetectorDescription/Parser/src/DDLBooleanSolid.h"
#include "DetectorDescription/Parser/src/DDLBox.h"
#include "DetectorDescription/Parser/src/DDLCompositeMaterial.h"
#include "DetectorDescription/Parser/src/DDLCone.h"
#include "DetectorDescription/Parser/src/DDLDivision.h"
#include "DetectorDescription/Parser/src/DDLElementaryMaterial.h"
#include "DetectorDescription/Parser/src/DDLEllipticalTube.h"
#include "DetectorDescription/Parser/src/DDLLogicalPart.h"
#include "DetectorDescription/Parser/src/DDLMap.h"
#include "DetectorDescription/Parser/src/DDLNumeric.h"
#include "DetectorDescription/Parser/src/DDLPolyGenerator.h"
#include "DetectorDescription/Parser/src/DDLPgonGenerator.h"
#include "DetectorDescription/Parser/src/DDLPosPart.h"
#include "DetectorDescription/Parser/src/DDLPseudoTrap.h"
#include "DetectorDescription/Parser/src/DDLRotationAndReflection.h"
#include "DetectorDescription/Parser/src/DDLRotationByAxis.h"
#include "DetectorDescription/Parser/src/DDLRotationSequence.h"
#include "DetectorDescription/Parser/src/DDLShapelessSolid.h"
#include "DetectorDescription/Parser/src/DDLSpecPar.h"
#include "DetectorDescription/Parser/src/DDLSphere.h"
#include "DetectorDescription/Parser/src/DDLString.h"
#include "DetectorDescription/Parser/src/DDLTorus.h"
#include "DetectorDescription/Parser/src/DDLTrapezoid.h"
#include "DetectorDescription/Parser/src/DDLTubs.h"
#include "DetectorDescription/Parser/src/DDLVector.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cstddef>
#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

DDLElementRegistry::DDLElementRegistry(void) {}

DDLElementRegistry::~DDLElementRegistry(void) { registry_.clear(); }

std::shared_ptr<DDXMLElement> DDLElementRegistry::getElement(const std::string& name) {
  RegistryMap::iterator it = registry_.find(name);
  std::shared_ptr<DDXMLElement> myret(nullptr);
  if (it != registry_.end()) {
    return it->second;
  } else {
    // Make the Solid handlers and register them.
    if (name == "Box") {
      myret = std::make_shared<DDLBox>(this);
    } else if (name == "Cone") {
      myret = std::make_shared<DDLCone>(this);
    } else if (name == "Polyhedra" || name == "Polycone") {
      myret = std::make_shared<DDLPolyGenerator>(this);
    } else if (name == "Trapezoid" || name == "Trd1") {
      myret = std::make_shared<DDLTrapezoid>(this);
    } else if (name == "PseudoTrap") {
      myret = std::make_shared<DDLPseudoTrap>(this);
    } else if (name == "Tubs" || name == "CutTubs" || name == "Tube" || name == "TruncTubs") {
      myret = std::make_shared<DDLTubs>(this);
    } else if (name == "Torus") {
      myret = std::make_shared<DDLTorus>(this);
    } else if (name == "UnionSolid" || name == "SubtractionSolid" || name == "IntersectionSolid") {
      myret = std::make_shared<DDLBooleanSolid>(this);
    } else if (name == "ShapelessSolid") {
      myret = std::make_shared<DDLShapelessSolid>(this);
    } else if (name == "Sphere") {
      myret = std::make_shared<DDLSphere>(this);
    } else if (name == "EllipticalTube") {
      myret = std::make_shared<DDLEllipticalTube>(this);
    } else if (name == "ExtrudedPolygon") {
      myret = std::make_shared<DDLPgonGenerator>(this);
    } else if (name == "Assembly")
      myret = std::make_shared<DDLAssembly>(this);

    //  LogicalParts, Positioners, Materials, Rotations, Reflections
    //  and Specific (Specified?) Parameters
    else if (name == "PosPart") {
      myret = std::make_shared<DDLPosPart>(this);
    } else if (name == "CompositeMaterial") {
      myret = std::make_shared<DDLCompositeMaterial>(this);
    } else if (name == "ElementaryMaterial") {
      myret = std::make_shared<DDLElementaryMaterial>(this);
    } else if (name == "LogicalPart") {
      myret = std::make_shared<DDLLogicalPart>(this);
    } else if (name == "ReflectionRotation" || name == "Rotation") {
      myret = std::make_shared<DDLRotationAndReflection>(this);
    } else if (name == "SpecPar") {
      myret = std::make_shared<DDLSpecPar>(this);
    } else if (name == "RotationSequence") {
      myret = std::make_shared<DDLRotationSequence>(this);
    } else if (name == "RotationByAxis") {
      myret = std::make_shared<DDLRotationByAxis>(this);
    }
    // Special, need them around.
    else if (name == "SpecParSection") {
      myret = std::make_shared<DDXMLElement>(this, true);
    } else if (name == "Vector") {
      myret = std::make_shared<DDLVector>(this);
    } else if (name == "Map") {
      myret = std::make_shared<DDLMap>(this);
    } else if (name == "String") {
      myret = std::make_shared<DDLString>(this);
    } else if (name == "Numeric") {
      myret = std::make_shared<DDLNumeric>(this);
    } else if (name == "Algorithm") {
      myret = std::make_shared<DDLAlgorithm>(this);
    } else if (name == "Division") {
      myret = std::make_shared<DDLDivision>(this);
    }

    // Supporting Cast of elements.
    //  All elements which simply accumulate attributes which are then used
    //  by one of the above elements.
    else if (name == "MaterialFraction" || name == "RZPoint" || name == "XYPoint" || name == "PartSelector" ||
             name == "Parameter" || name == "ZSection" || name == "ZXYSection" || name == "Translation" ||
             name == "rSolid" || name == "rMaterial" || name == "rParent" || name == "rChild" || name == "rRotation" ||
             name == "rReflectionRotation" || name == "DDDefinition") {
      myret = std::make_shared<DDXMLElement>(this);
    }

    //  IF it is a new element return a default XMLElement which processes nothing.
    //  Since there are elements in the XML which require no processing, they
    //  can all use the same DDXMLElement which defaults to anything.  A validated
    //  XML document (i.e. validated with XML Schema) will work properly.
    //  As of 8/16/2002:  Elements like LogicalPartSection and any other *Section
    //  XML elements of the DDLSchema are taken care of by this.
    else {
      myret = std::make_shared<DDXMLElement>(this);
    }

    // Actually register the thing
    registry_[name] = myret;
  }
  return myret;
}
