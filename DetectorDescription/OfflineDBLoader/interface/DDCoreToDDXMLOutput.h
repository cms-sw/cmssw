#ifndef GUARD_DDCoreToDDXMLOutput_H
#define GUARD_DDCoreToDDXMLOutput_H

#include <iostream>
#include <set>
#include <string>
#include <utility>

#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDPosData.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"

#include "DetectorDescription/DDCMS/interface/DDParsingContext.h"

#include "DD4hep/SpecParRegistry.h"

#include "TGeoMaterial.h"

class DDPartSelection;
class DDRotation;
struct DDPosData;

/** @class DDCoreToDDXMLOutput DDCoreToDDXMLOutput.h
 *
 *  @author:  Michael Case  Revison for DD4hep
 *  @version: 1.0
 * 
 *  Description:
 *       The idea is to store one large XML file containing the XML elements
 *     in DDL (Detector Description Language).  In doing so, all DDAlgorithms
 *     or other code generated DD in-memory components should be reproduced 
 *     without the need for them to run again if the geometry is read into
 *     DDLParser again.
 */

struct DDCoreToDDXMLOutput {
  void solid(const DDSolid& solid, std::ostream& xos);
  void solid(const dd4hep::Solid& solid, const cms::DDParsingContext& context, std::ostream& xos);

  void element(const TGeoMaterial* element, std::ostream& xos);
  void material(const DDMaterial& material, std::ostream& xos);
  void material(
      const std::pair<std::string, std::pair<double, std::vector<cms::DDParsingContext::CompositeMaterial>>>& material,
      std::ostream& xos);

  void rotation(const DDRotation& rotation, std::ostream& xos, const std::string& rotn = "");
  void rotation(const dd4hep::Rotation3D& rotation,
                std::ostream& xos,
                const cms::DDParsingContext& context,
                const std::string& rotn);

  void logicalPart(const DDLogicalPart& lp, std::ostream& xos);
  void logicalPart(const std::string& asName, std::ostream& xos);
  void logicalPart(const TGeoVolume& lp, std::ostream& xos);

  void position(const DDLogicalPart& parent,
                const DDLogicalPart& child,
                DDPosData* edgeToChild,
                int& rotNameSeed,
                std::ostream& xos);
  void position(const TGeoVolume& parent,
                const TGeoNode& child,
                const std::string& childName,
                cms::DDParsingContext& context,
                std::ostream& xos);
  //
  // left in for now as legacy...
  void specpar(const DDSpecifics& sp, std::ostream& xos);
  void specpar(const std::pair<DDsvalues_type, std::set<const DDPartSelection*>>& pssv, std::ostream& xos);
  void specpar(const std::string& name, const dd4hep::SpecPar& specPar, std::ostream& xos);

  static std::string trimShapeName(const std::string& solidName);
  std::string ns_;  // default namespace
};

#endif
