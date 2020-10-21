#include "DD4hep/DetFactoryHelper.h"
#include "DD4hep/DetectorHelper.h"
#include "DD4hep/DD4hepUnits.h"
#include "DD4hep/GeoHandler.h"
#include "DD4hep/Printout.h"
#include "DD4hep/Plugins.h"
#include "DD4hep/detail/SegmentationsInterna.h"
#include "DD4hep/detail/DetectorInterna.h"
#include "DD4hep/detail/ObjectsInterna.h"
#include "DD4hep/MatrixHelpers.h"

#include "XML/Utilities.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDAlgoArguments.h"
#include "DetectorDescription/DDCMS/interface/DDNamespace.h"
#include "DetectorDescription/DDCMS/interface/DDParsingContext.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"

#include "TGeoManager.h"
#include "TGeoMaterial.h"

#include <climits>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>
#include <unordered_map>
#include <utility>

#define EDM_ML_DEBUG

using namespace std;
using namespace dd4hep;
using namespace cms;
using namespace cms_units::operators;

namespace dd4hep {

  namespace {

    atomic<UInt_t> unique_mat_id = 0xAFFEFEED;

    class include_constants;
    class include_load;
    class include_unload;
    class print_xml_doc;

    class ConstantsSection;
    class DDLConstant;

    struct DDRegistry {
      std::vector<xml::Document> includes;
      std::unordered_map<std::string, std::string> unresolvedConst;
      std::unordered_map<std::string, std::string> originalConst;
    };

    class MaterialSection;
    class DDLElementaryMaterial;
    class DDLCompositeMaterial;

    class RotationSection;
    class DDLRotation;
    class DDLReflectionRotation;
    class DDLRotationSequence;
    class DDLRotationByAxis;
    class DDLTransform3D;

    class PosPartSection;
    class DDLPosPart;
    class DDLDivision;

    class LogicalPartSection;
    class DDLLogicalPart;

    class SolidSection;
    class DDLExtrudedPolygon;
    class DDLShapeless;
    class DDLTrapezoid;
    class DDLEllipticalTube;
    class DDLPseudoTrap;
    class DDLPolyhedra;
    class DDLPolycone;
    class DDLTorus;
    class DDLTrd1;
    class DDLTrd2;
    class DDLTruncTubs;
    class DDLCutTubs;
    class DDLTubs;
    class DDLBox;
    class DDLCone;
    class DDLSphere;
    class DDLUnionSolid;
    class DDLIntersectionSolid;
    class DDLSubtractionSolid;

    class DDLAlgorithm;
    class DDLVector;

    class SpecParSection;
    class DDLSpecPar;
    class PartSelector;
    class Parameter;

    class debug;
  }  // namespace

  TGeoCombiTrans* createPlacement(const Rotation3D& iRot, const Position& iTrans) {
    double elements[9];
    iRot.GetComponents(elements);
    TGeoRotation r;
    r.SetMatrix(elements);

    TGeoTranslation t(iTrans.x(), iTrans.y(), iTrans.z());

    return new TGeoCombiTrans(t, r);
  }

  /// Converter instances implemented in this compilation unit
  template <>
  void Converter<debug>::operator()(xml_h element) const;
  template <>
  void Converter<print_xml_doc>::operator()(xml_h element) const;

  /// Converter for <ConstantsSection/> tags
  template <>
  void Converter<ConstantsSection>::operator()(xml_h element) const;
  template <>
  void Converter<DDLConstant>::operator()(xml_h element) const;
  template <>
  void Converter<DDRegistry>::operator()(xml_h element) const;

  /// Converter for <MaterialSection/> tags
  template <>
  void Converter<MaterialSection>::operator()(xml_h element) const;
  template <>
  void Converter<DDLElementaryMaterial>::operator()(xml_h element) const;
  template <>
  void Converter<DDLCompositeMaterial>::operator()(xml_h element) const;

  /// Converter for <RotationSection/> tags
  template <>
  void Converter<RotationSection>::operator()(xml_h element) const;
  /// Converter for <DDLRotation/> tags
  template <>
  void Converter<DDLRotation>::operator()(xml_h element) const;
  /// Converter for <DDLReflectionRotation/> tags
  template <>
  void Converter<DDLReflectionRotation>::operator()(xml_h element) const;
  /// Converter for <DDLRotationSequence/> tags
  template <>
  void Converter<DDLRotationSequence>::operator()(xml_h element) const;
  /// Converter for <DDLRotationByAxis/> tags
  template <>
  void Converter<DDLRotationByAxis>::operator()(xml_h element) const;
  template <>
  void Converter<DDLTransform3D>::operator()(xml_h element) const;

  /// Generic converter for <LogicalPartSection/> tags
  template <>
  void Converter<LogicalPartSection>::operator()(xml_h element) const;
  template <>
  void Converter<DDLLogicalPart>::operator()(xml_h element) const;

  /// Generic converter for <PosPartSection/> tags
  template <>
  void Converter<PosPartSection>::operator()(xml_h element) const;
  /// Converter for <PosPart/> tags
  template <>
  void Converter<DDLPosPart>::operator()(xml_h element) const;
  /// Converter for <Division/> tags
  template <>
  void Converter<DDLDivision>::operator()(xml_h element) const;

  /// Generic converter for <SpecParSection/> tags
  template <>
  void Converter<SpecParSection>::operator()(xml_h element) const;
  template <>
  void Converter<DDLSpecPar>::operator()(xml_h element) const;
  template <>
  void Converter<PartSelector>::operator()(xml_h element) const;
  template <>
  void Converter<Parameter>::operator()(xml_h element) const;

  /// Generic converter for solids: <SolidSection/> tags
  template <>
  void Converter<SolidSection>::operator()(xml_h element) const;
  /// Converter for <UnionSolid/> tags
  template <>
  void Converter<DDLUnionSolid>::operator()(xml_h element) const;
  /// Converter for <SubtractionSolid/> tags
  template <>
  void Converter<DDLSubtractionSolid>::operator()(xml_h element) const;
  /// Converter for <IntersectionSolid/> tags
  template <>
  void Converter<DDLIntersectionSolid>::operator()(xml_h element) const;
  /// Converter for <PseudoTrap/> tags
  template <>
  void Converter<DDLPseudoTrap>::operator()(xml_h element) const;
  /// Converter for <ExtrudedPolygon/> tags
  template <>
  void Converter<DDLExtrudedPolygon>::operator()(xml_h element) const;
  /// Converter for <ShapelessSolid/> tags
  template <>
  void Converter<DDLShapeless>::operator()(xml_h element) const;
  /// Converter for <Trapezoid/> tags
  template <>
  void Converter<DDLTrapezoid>::operator()(xml_h element) const;
  /// Converter for <Polycone/> tags
  template <>
  void Converter<DDLPolycone>::operator()(xml_h element) const;
  /// Converter for <Polyhedra/> tags
  template <>
  void Converter<DDLPolyhedra>::operator()(xml_h element) const;
  /// Converter for <EllipticalTube/> tags
  template <>
  void Converter<DDLEllipticalTube>::operator()(xml_h element) const;
  /// Converter for <Torus/> tags
  template <>
  void Converter<DDLTorus>::operator()(xml_h element) const;
  /// Converter for <Tubs/> tags
  template <>
  void Converter<DDLTubs>::operator()(xml_h element) const;
  /// Converter for <CutTubs/> tags
  template <>
  void Converter<DDLCutTubs>::operator()(xml_h element) const;
  /// Converter for <TruncTubs/> tags
  template <>
  void Converter<DDLTruncTubs>::operator()(xml_h element) const;
  /// Converter for <Sphere/> tags
  template <>
  void Converter<DDLSphere>::operator()(xml_h element) const;
  /// Converter for <Trd1/> tags
  template <>
  void Converter<DDLTrd1>::operator()(xml_h element) const;
  /// Converter for <Trd2/> tags
  template <>
  void Converter<DDLTrd2>::operator()(xml_h element) const;
  /// Converter for <Cone/> tags
  template <>
  void Converter<DDLCone>::operator()(xml_h element) const;
  /// Converter for <DDLBox/> tags
  template <>
  void Converter<DDLBox>::operator()(xml_h element) const;
  /// Converter for <Algorithm/> tags
  template <>
  void Converter<DDLAlgorithm>::operator()(xml_h element) const;
  /// Converter for <Vector/> tags
  template <>
  void Converter<DDLVector>::operator()(xml_h element) const;

  /// DD4hep specific: Load include file
  template <>
  void Converter<include_load>::operator()(xml_h element) const;
  /// DD4hep specific: Unload include file
  template <>
  void Converter<include_unload>::operator()(xml_h element) const;
  /// DD4hep specific: Process constants objects
  template <>
  void Converter<include_constants>::operator()(xml_h element) const;
}  // namespace dd4hep

/// Converter for <ConstantsSection/> tags
template <>
void Converter<ConstantsSection>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>(), element);
  cms::DDParsingContext* const context = ns.context();
  xml_coll_t(element, DD_CMU(Constant)).for_each(Converter<DDLConstant>(description, context, optional));
  xml_coll_t(element, DD_CMU(Vector)).for_each(Converter<DDLVector>(description, context, optional));
}

/// Converter for <MaterialSection/> tags
template <>
void Converter<MaterialSection>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>(), element);
  xml_coll_t(element, DD_CMU(ElementaryMaterial))
      .for_each(Converter<DDLElementaryMaterial>(description, ns.context(), optional));
  xml_coll_t(element, DD_CMU(CompositeMaterial))
      .for_each(Converter<DDLCompositeMaterial>(description, ns.context(), optional));
}

template <>
void Converter<RotationSection>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>(), element);
  xml_coll_t(element, DD_CMU(Rotation)).for_each(Converter<DDLRotation>(description, ns.context(), optional));
  xml_coll_t(element, DD_CMU(ReflectionRotation))
      .for_each(Converter<DDLReflectionRotation>(description, ns.context(), optional));
  xml_coll_t(element, DD_CMU(RotationSequence))
      .for_each(Converter<DDLRotationSequence>(description, ns.context(), optional));
  xml_coll_t(element, DD_CMU(RotationByAxis))
      .for_each(Converter<DDLRotationByAxis>(description, ns.context(), optional));
}

template <>
void Converter<PosPartSection>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>(), element);
  xml_coll_t(element, DD_CMU(Division)).for_each(Converter<DDLDivision>(description, ns.context(), optional));
  xml_coll_t(element, DD_CMU(PosPart)).for_each(Converter<DDLPosPart>(description, ns.context(), optional));
  xml_coll_t(element, DD_CMU(Algorithm)).for_each(Converter<DDLAlgorithm>(description, ns.context(), optional));
}

template <>
void Converter<SpecParSection>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>(), element);
  xml_coll_t(element, DD_CMU(SpecPar)).for_each(Converter<DDLSpecPar>(description, ns.context(), optional));
}

template <>
void Converter<DDLSpecPar>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>(), element);
  xml_coll_t(element, DD_CMU(PartSelector)).for_each(Converter<PartSelector>(description, ns.context(), optional));
  xml_coll_t(element, DD_CMU(Parameter)).for_each(Converter<Parameter>(description, ns.context(), optional));
}

/// Generic converter for  <LogicalPartSection/> tags
template <>
void Converter<LogicalPartSection>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>(), element);
  xml_coll_t(element, DD_CMU(LogicalPart)).for_each(Converter<DDLLogicalPart>(description, ns.context(), optional));
}

/// Generic converter for  <SolidSection/> tags
template <>
void Converter<SolidSection>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>(), element);
  for (xml_coll_t solid(element, _U(star)); solid; ++solid) {
    using cms::hash;
    switch (hash(solid.tag())) {
      case hash("Box"):
        Converter<DDLBox>(description, ns.context(), optional)(solid);
        break;
      case hash("Polycone"):
        Converter<DDLPolycone>(description, ns.context(), optional)(solid);
        break;
      case hash("Polyhedra"):
        Converter<DDLPolyhedra>(description, ns.context(), optional)(solid);
        break;
      case hash("Tubs"):
        Converter<DDLTubs>(description, ns.context(), optional)(solid);
        break;
      case hash("CutTubs"):
        Converter<DDLCutTubs>(description, ns.context(), optional)(solid);
        break;
      case hash("TruncTubs"):
        Converter<DDLTruncTubs>(description, ns.context(), optional)(solid);
        break;
      case hash("Tube"):
        Converter<DDLTubs>(description, ns.context(), optional)(solid);
        break;
      case hash("Trd1"):
        Converter<DDLTrd1>(description, ns.context(), optional)(solid);
        break;
      case hash("Trd2"):
        Converter<DDLTrd2>(description, ns.context(), optional)(solid);
        break;
      case hash("Cone"):
        Converter<DDLCone>(description, ns.context(), optional)(solid);
        break;
      case hash("Sphere"):
        Converter<DDLSphere>(description, ns.context(), optional)(solid);
        break;
      case hash("EllipticalTube"):
        Converter<DDLEllipticalTube>(description, ns.context(), optional)(solid);
        break;
      case hash("Torus"):
        Converter<DDLTorus>(description, ns.context(), optional)(solid);
        break;
      case hash("PseudoTrap"):
        Converter<DDLPseudoTrap>(description, ns.context(), optional)(solid);
        break;
      case hash("ExtrudedPolygon"):
        Converter<DDLExtrudedPolygon>(description, ns.context(), optional)(solid);
        break;
      case hash("Trapezoid"):
        Converter<DDLTrapezoid>(description, ns.context(), optional)(solid);
        break;
      case hash("UnionSolid"):
        Converter<DDLUnionSolid>(description, ns.context(), optional)(solid);
        break;
      case hash("SubtractionSolid"):
        Converter<DDLSubtractionSolid>(description, ns.context(), optional)(solid);
        break;
      case hash("IntersectionSolid"):
        Converter<DDLIntersectionSolid>(description, ns.context(), optional)(solid);
        break;
      case hash("ShapelessSolid"):
        Converter<DDLShapeless>(description, ns.context(), optional)(solid);
        break;
      default:
        throw std::runtime_error("Request to process unknown shape '" + xml_dim_t(solid).nameStr() + "' [" +
                                 solid.tag() + "]");
        break;
    }
  }
}

/// Converter for <Constant/> tags
template <>
void Converter<DDLConstant>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  DDRegistry* res = _option<DDRegistry>();
  xml_dim_t constant = element;
  xml_dim_t par = constant.parent();
  bool eval = par.hasAttr(_U(eval)) ? par.attr<bool>(_U(eval)) : false;
  string val = constant.valueStr();
  string nam = constant.nameStr();
  string real = ns.prepend(nam);
  string typ = eval ? "number" : "string";
  size_t idx = val.find('[');

  if (constant.hasAttr(_U(type)))
    typ = constant.typeStr();

  if (idx == string::npos || typ == "string") {
    try {
      ns.addConstant(nam, val, typ);
      res->originalConst[real] = val;
    } catch (const exception& e) {
#ifdef EDM_ML_DEBUG

      printout(INFO,
               "DD4CMS",
               "++ Unresolved constant: %s = %s [%s]. Try to resolve later. [%s]",
               real.c_str(),
               val.c_str(),
               typ.c_str(),
               e.what());
#endif
    }
    return;
  }
  // Setup the resolution mechanism in Converter<resolve>
  while (idx != string::npos) {
    ++idx;
    size_t idp = val.find(':', idx);
    size_t idq = val.find(']', idx);
    if (idp == string::npos || idp > idq)
      val.insert(idx, ns.name());
    else if (idp != string::npos && idp < idq)
      val[idp] = NAMESPACE_SEP;
    idx = val.find('[', idx);
  }

#ifdef EDM_ML_DEBUG

  printout(
      ns.context()->debug_constants ? ALWAYS : DEBUG, "Constant", "Unresolved: %s -> %s", real.c_str(), val.c_str());

#endif

  res->originalConst[real] = val;
  res->unresolvedConst[real] = val;
}

/// Converter for <DDLElementaryMaterial/> tags
template <>
void Converter<DDLElementaryMaterial>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  xml_dim_t xmat(element);
  string nam = ns.prepend(xmat.nameStr());
  TGeoManager& mgr = description.manager();
  TGeoMaterial* mat = mgr.GetMaterial(nam.c_str());
  if (nullptr == mat) {
    const char* matname = nam.c_str();
    double density = xmat.attr<double>(DD_CMU(density)) / (dd4hep::g / dd4hep::cm3);
    int atomicNumber = xmat.attr<int>(DD_CMU(atomicNumber));
    double atomicWeight = xmat.attr<double>(DD_CMU(atomicWeight)) / (dd4hep::g / dd4hep::mole);
    TGeoElementTable* tab = mgr.GetElementTable();
    int nElem = tab->GetNelements();

#ifdef EDM_ML_DEBUG

    printout(ns.context()->debug_materials ? ALWAYS : DEBUG, "DD4CMS", "+++ Element table size = %d", nElem);

#endif

    if (nElem <= 1) {  // Restore the element table DD4hep destroyed.
      tab->TGeoElementTable::~TGeoElementTable();
      new (tab) TGeoElementTable();
      tab->BuildDefaultElements();
    }
    TGeoMixture* mix = new TGeoMixture(nam.c_str(), 1, density);
    TGeoElement* elt = tab->FindElement(xmat.nameStr().c_str());

#ifdef EDM_ML_DEBUG

    printout(ns.context()->debug_materials ? ALWAYS : DEBUG,
             "DD4CMS",
             "+++ Searching for material %-48s  elt_ptr = %ld",
             xmat.nameStr().c_str(),
             elt);

    printout(ns.context()->debug_materials ? ALWAYS : DEBUG,
             "DD4CMS",
             "+++ Converting material %-48s  Atomic weight %8.3f   [g/mol], Atomic number %u, Density: %8.3f [g/cm3] "
             "ROOT: %8.3f [g/cm3]",
             ('"' + nam + '"').c_str(),
             atomicWeight,
             atomicNumber,
             density,
             mix->GetDensity());

#endif

    bool newMatDef = false;

    if (elt) {
      // A is Mass of a mole in Geant4 units for atoms with atomic shell

#ifdef EDM_ML_DEBUG

      printout(ns.context()->debug_materials ? ALWAYS : DEBUG,
               "DD4CMS",
               "    ROOT definition of %-50s Atomic weight %g, Atomic number %u, Number of nucleons %u",
               elt->GetName(),
               elt->A(),
               elt->Z(),
               elt->N());
      printout(ns.context()->debug_materials ? ALWAYS : DEBUG,
               "DD4CMS",
               "+++ Compared to XML values: Atomic weight %g, Atomic number %u",
               atomicWeight,
               atomicNumber);
#endif

      static constexpr double const weightTolerance = 1.0e-6;
      if (atomicNumber != elt->Z() ||
          (std::abs(atomicWeight - elt->A()) > (weightTolerance * (atomicWeight + elt->A()))))
        newMatDef = true;
    }

    if (!elt || newMatDef) {
      if (newMatDef) {
#ifdef EDM_ML_DEBUG

        printout(ns.context()->debug_materials ? ALWAYS : DEBUG,
                 "DD4CMS Warning",
                 "+++ Converter<ElementaryMaterial> Different definition of a default element with name:%s [CREATE NEW "
                 "MATERIAL]",
                 matname);

#endif

      } else {
#ifdef EDM_ML_DEBUG

        printout(ns.context()->debug_materials ? ALWAYS : DEBUG,
                 "DD4CMS Warning",
                 "+++ Converter<ElementaryMaterial> No default element present with name:%s  [CREATE NEW MATERIAL]",
                 matname);

#endif
      }
      elt = new TGeoElement(xmat.nameStr().c_str(), "CMS element", atomicNumber, atomicWeight);
    }

    mix->AddElement(elt, 1.0);

    /// Create medium from the material
    TGeoMedium* medium = mgr.GetMedium(matname);
    if (nullptr == medium) {
      --unique_mat_id;
      medium = new TGeoMedium(matname, unique_mat_id, mix);
      medium->SetTitle("material");
      medium->SetUniqueID(unique_mat_id);
    }
  }
}

/// Converter for <DDLCompositeMaterial/> tags
template <>
void Converter<DDLCompositeMaterial>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  xml_dim_t xmat(element);
  string nam = ns.prepend(xmat.nameStr());

  TGeoManager& mgr = description.manager();
  TGeoMaterial* mat = mgr.GetMaterial(nam.c_str());
  if (nullptr == mat) {
    const char* matname = nam.c_str();
    double density = xmat.attr<double>(DD_CMU(density)) / (dd4hep::g / dd4hep::cm3);
    xml_coll_t composites(xmat, DD_CMU(MaterialFraction));
    TGeoMixture* mix = new TGeoMixture(nam.c_str(), composites.size(), density);

#ifdef EDM_ML_DEBUG

    printout(ns.context()->debug_materials ? ALWAYS : DEBUG,
             "DD4CMS",
             "++ Converting material %-48s  Density: %8.3f [g/cm3] ROOT: %8.3f [g/cm3]",
             ('"' + nam + '"').c_str(),
             density,
             mix->GetDensity());

#endif

    for (composites.reset(); composites; ++composites) {
      xml_dim_t xfrac(composites);
      xml_dim_t xfrac_mat(xfrac.child(DD_CMU(rMaterial)));
      double fraction = xfrac.fraction();
      string fracname = ns.realName(xfrac_mat.nameStr());

      TGeoMaterial* frac_mat = mgr.GetMaterial(fracname.c_str());
      if (frac_mat == nullptr)  // Try to find it within this namespace
        frac_mat = mgr.GetMaterial(ns.prepend(fracname).c_str());
      if (frac_mat) {
        mix->AddElement(frac_mat, fraction);
        continue;
      }

#ifdef EDM_ML_DEBUG

      printout(ns.context()->debug_materials ? ALWAYS : DEBUG,
               "DD4CMS Warning",
               "+++ Composite material \"%s\" [nor \"%s\"] not present! [delay resolution]",
               fracname.c_str(),
               ns.prepend(fracname).c_str());

#endif

      ns.context()->unresolvedMaterials[nam].emplace_back(
          cms::DDParsingContext::CompositeMaterial(ns.prepend(fracname), fraction));
    }
    mix->SetRadLen(0e0);
    /// Create medium from the material
    TGeoMedium* medium = mgr.GetMedium(matname);
    if (nullptr == medium) {
      --unique_mat_id;
      medium = new TGeoMedium(matname, unique_mat_id, mix);
      medium->SetTitle("material");
      medium->SetUniqueID(unique_mat_id);
    }
  }
}

/// Converter for <Rotation/> tags
template <>
void Converter<DDLRotation>::operator()(xml_h element) const {
  cms::DDParsingContext* context = _param<cms::DDParsingContext>();
  cms::DDNamespace ns(context);
  xml_dim_t xrot(element);
  string nam = xrot.nameStr();
  double thetaX = xrot.hasAttr(DD_CMU(thetaX)) ? ns.attr<double>(xrot, DD_CMU(thetaX)) : 0e0;
  double phiX = xrot.hasAttr(DD_CMU(phiX)) ? ns.attr<double>(xrot, DD_CMU(phiX)) : 0e0;
  double thetaY = xrot.hasAttr(DD_CMU(thetaY)) ? ns.attr<double>(xrot, DD_CMU(thetaY)) : 0e0;
  double phiY = xrot.hasAttr(DD_CMU(phiY)) ? ns.attr<double>(xrot, DD_CMU(phiY)) : 0e0;
  double thetaZ = xrot.hasAttr(DD_CMU(thetaZ)) ? ns.attr<double>(xrot, DD_CMU(thetaZ)) : 0e0;
  double phiZ = xrot.hasAttr(DD_CMU(phiZ)) ? ns.attr<double>(xrot, DD_CMU(phiZ)) : 0e0;
  Rotation3D rot = makeRotation3D(thetaX, phiX, thetaY, phiY, thetaZ, phiZ);

#ifdef EDM_ML_DEBUG

  printout(context->debug_rotations ? ALWAYS : DEBUG,
           "DD4CMS",
           "+++ Adding rotation: %-32s: (theta/phi)[rad] X: %6.3f %6.3f Y: %6.3f %6.3f Z: %6.3f %6.3f",
           ns.prepend(nam).c_str(),
           thetaX,
           phiX,
           thetaY,
           phiY,
           thetaZ,
           phiZ);

#endif

  ns.addRotation(nam, rot);
}

/// Converter for <ReflectionRotation/> tags
template <>
void Converter<DDLReflectionRotation>::operator()(xml_h element) const {
  cms::DDParsingContext* context = _param<cms::DDParsingContext>();
  cms::DDNamespace ns(context);
  xml_dim_t xrot(element);
  string name = xrot.nameStr();
  double thetaX = xrot.hasAttr(DD_CMU(thetaX)) ? ns.attr<double>(xrot, DD_CMU(thetaX)) : 0e0;
  double phiX = xrot.hasAttr(DD_CMU(phiX)) ? ns.attr<double>(xrot, DD_CMU(phiX)) : 0e0;
  double thetaY = xrot.hasAttr(DD_CMU(thetaY)) ? ns.attr<double>(xrot, DD_CMU(thetaY)) : 0e0;
  double phiY = xrot.hasAttr(DD_CMU(phiY)) ? ns.attr<double>(xrot, DD_CMU(phiY)) : 0e0;
  double thetaZ = xrot.hasAttr(DD_CMU(thetaZ)) ? ns.attr<double>(xrot, DD_CMU(thetaZ)) : 0e0;
  double phiZ = xrot.hasAttr(DD_CMU(phiZ)) ? ns.attr<double>(xrot, DD_CMU(phiZ)) : 0e0;

#ifdef EDM_ML_DEBUG

  printout(context->debug_rotations ? ALWAYS : DEBUG,
           "DD4CMS",
           "+++ Adding reflection rotation: %-32s: (theta/phi)[rad] X: %6.3f %6.3f Y: %6.3f %6.3f Z: %6.3f %6.3f",
           ns.prepend(name).c_str(),
           thetaX,
           phiX,
           thetaY,
           phiY,
           thetaZ,
           phiZ);

#endif

  Rotation3D rot = makeRotReflect(thetaX, phiX, thetaY, phiY, thetaZ, phiZ);
  ns.addRotation(name, rot);
}

/// Converter for <RotationSequence/> tags
template <>
void Converter<DDLRotationSequence>::operator()(xml_h element) const {
  cms::DDParsingContext* context = _param<cms::DDParsingContext>();
  cms::DDNamespace ns(context);
  xml_dim_t xrot(element);
  string nam = xrot.nameStr();
  Rotation3D rot;
  xml_coll_t rotations(xrot, DD_CMU(RotationByAxis));
  for (rotations.reset(); rotations; ++rotations) {
    string axis = ns.attr<string>(rotations, DD_CMU(axis));
    double angle = ns.attr<double>(rotations, _U(angle));
    rot = makeRotation3D(rot, axis, angle);

#ifdef EDM_ML_DEBUG

    printout(context->debug_rotations ? ALWAYS : DEBUG,
             "DD4CMS",
             "+   Adding rotation to: %-29s: (axis/angle)[rad] Axis: %s Angle: %6.3f",
             nam.c_str(),
             axis.c_str(),
             angle);

#endif
  }
  double xx, xy, xz;
  double yx, yy, yz;
  double zx, zy, zz;
  rot.GetComponents(xx, xy, xz, yx, yy, yz, zx, zy, zz);

#ifdef EDM_ML_DEBUG

  printout(context->debug_rotations ? ALWAYS : DEBUG,
           "DD4CMS",
           "+++ Adding rotation sequence: %-23s: %6.3f %6.3f %6.3f, %6.3f %6.3f %6.3f, %6.3f %6.3f %6.3f",
           ns.prepend(nam).c_str(),
           xx,
           xy,
           xz,
           yx,
           yy,
           yz,
           zx,
           zy,
           zz);

#endif

  ns.addRotation(nam, rot);
}

/// Converter for <RotationByAxis/> tags
template <>
void Converter<DDLRotationByAxis>::operator()(xml_h element) const {
  cms::DDParsingContext* context = _param<cms::DDParsingContext>();
  cms::DDNamespace ns(context);
  xml_dim_t xrot(element);
  xml_dim_t par(xrot.parent());
  if (xrot.hasAttr(_U(name))) {
    string nam = xrot.nameStr();
    string axis = ns.attr<string>(xrot, DD_CMU(axis));
    double angle = ns.attr<double>(xrot, _U(angle));
    Rotation3D rot;
    rot = makeRotation3D(rot, axis, angle);

#ifdef EDM_ML_DEBUG

    printout(context->debug_rotations ? ALWAYS : DEBUG,
             "DD4CMS",
             "+++ Adding rotation: %-32s: (axis/angle)[rad] Axis: %s Angle: %6.3f",
             ns.prepend(nam).c_str(),
             axis.c_str(),
             angle);

#endif

    ns.addRotation(nam, rot);
  }
}

/// Converter for <LogicalPart/> tags
template <>
void Converter<DDLLogicalPart>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string sol = e.child(DD_CMU(rSolid)).attr<string>(_U(name));
  string mat = e.child(DD_CMU(rMaterial)).attr<string>(_U(name));
  string volName = ns.prepend(e.attr<string>(_U(name)));
  Solid solid = ns.solid(sol);
  Material material = ns.material(mat);

#ifdef EDM_ML_DEBUG
  Volume volume =
#endif

      ns.addVolume(Volume(volName, solid, material));

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_volumes ? ALWAYS : DEBUG,
           "DD4CMS",
           "+++ %s Volume: %-24s [%s] Shape: %-32s [%s] Material: %-40s [%s]",
           e.tag().c_str(),
           volName.c_str(),
           volume.isValid() ? "VALID" : "INVALID",
           sol.c_str(),
           solid.isValid() ? "VALID" : "INVALID",
           mat.c_str(),
           material.isValid() ? "VALID" : "INVALID");

#endif
}

/// Helper converter
template <>
void Converter<DDLTransform3D>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  Transform3D* tr = _option<Transform3D>();
  xml_dim_t e(element);
  xml_dim_t translation = e.child(DD_CMU(Translation), false);
  xml_dim_t rotation = e.child(DD_CMU(Rotation), false);
  xml_dim_t refRotation = e.child(DD_CMU(rRotation), false);
  xml_dim_t refReflectionRotation = e.child(DD_CMU(rReflectionRotation), false);
  Position pos;
  Rotation3D rot;

  if (translation.ptr()) {
    double x = ns.attr<double>(translation, _U(x));
    double y = ns.attr<double>(translation, _U(y));
    double z = ns.attr<double>(translation, _U(z));
    pos = Position(x, y, z);
  }
  if (rotation.ptr()) {
    double x = ns.attr<double>(rotation, _U(x));
    double y = ns.attr<double>(rotation, _U(y));
    double z = ns.attr<double>(rotation, _U(z));
    rot = RotationZYX(z, y, x);
  } else if (refRotation.ptr()) {
    string rotName = ns.prepend(refRotation.nameStr());
    rot = ns.rotation(rotName);
  } else if (refReflectionRotation.ptr()) {
    string rotName = ns.prepend(refReflectionRotation.nameStr());
    rot = ns.rotation(rotName);
  }
  *tr = Transform3D(rot, pos);
}

/// Converter for <PosPart/> tags
template <>
void Converter<DDLPosPart>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());  //, element, true );
  xml_dim_t e(element);
  int copy = e.attr<int>(DD_CMU(copyNumber));
  string parentName = ns.prepend(ns.attr<string>(e.child(DD_CMU(rParent)), _U(name)));
  string childName = ns.prepend(ns.attr<string>(e.child(DD_CMU(rChild)), _U(name)));
  Volume parent = ns.volume(parentName, false);
  Volume child = ns.volume(childName, false);

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_placements ? ALWAYS : DEBUG,
           "DD4CMS",
           "+++ %s Parent: %-24s [%s] Child: %-32s [%s] copy:%d",
           e.tag().c_str(),
           parentName.c_str(),
           parent.isValid() ? "VALID" : "INVALID",
           childName.c_str(),
           child.isValid() ? "VALID" : "INVALID",
           copy);

#endif

  if (!parent.isValid() && strchr(parentName.c_str(), NAMESPACE_SEP) == nullptr)
    parentName = ns.prepend(parentName);
  parent = ns.volume(parentName);

  if (!child.isValid() && strchr(childName.c_str(), NAMESPACE_SEP) == nullptr)
    childName = ns.prepend(childName);
  child = ns.volume(childName, false);

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_placements ? ALWAYS : DEBUG,
           "DD4CMS",
           "+++ %s Parent: %-24s [%s] Child: %-32s [%s] copy:%d",
           e.tag().c_str(),
           parentName.c_str(),
           parent.isValid() ? "VALID" : "INVALID",
           childName.c_str(),
           child.isValid() ? "VALID" : "INVALID",
           copy);

#endif

  PlacedVolume pv;
  if (child.isValid()) {
    Transform3D transform;
    Converter<DDLTransform3D>(description, param, &transform)(element);

    // FIXME: workaround for Reflection rotation
    // copy from DDCore/src/Volumes.cpp to replace
    // static PlacedVolume _addNode(TGeoVolume* par, TGeoVolume* daughter, int id, TGeoMatrix* transform)
    if (!parent) {
      except("dd4hep", "Volume: Attempt to assign daughters to an invalid physical parent volume.");
    }
    if (!child) {
      except("dd4hep", "Volume: Attempt to assign an invalid physical daughter volume.");
    }
    TGeoShape* shape = child->GetShape();
    // Need to fix the daughter's BBox of assemblies, if the BBox was not calculated....
    if (shape->IsA() == TGeoShapeAssembly::Class()) {
      TGeoShapeAssembly* as = (TGeoShapeAssembly*)shape;
      if (std::fabs(as->GetDX()) < numeric_limits<double>::epsilon() &&
          std::fabs(as->GetDY()) < numeric_limits<double>::epsilon() &&
          std::fabs(as->GetDZ()) < numeric_limits<double>::epsilon()) {
        as->NeedsBBoxRecompute();
        as->ComputeBBox();
      }
    }
    TGeoNode* n;
    TString nam_id = TString::Format("%s_%d", child->GetName(), copy);
    n = static_cast<TGeoNode*>(parent->GetNode(nam_id));
    if (n != nullptr) {
      printout(ERROR, "PlacedVolume", "++ Attempt to add already exiting node %s", (const char*)nam_id);
    }

    Rotation3D rot(transform.Rotation());
    Translation3D trans(transform.Translation());
    double x, y, z;
    trans.GetComponents(x, y, z);
    Position pos(x, y, z);
    parent->AddNode(child, copy, createPlacement(rot, pos));

    n = static_cast<TGeoNode*>(parent->GetNode(nam_id));
    n->TGeoNode::SetUserExtension(new PlacedVolume::Object());
    pv = PlacedVolume(n);
  }
  if (!pv.isValid()) {
    printout(ERROR,
             "DD4CMS",
             "+++ Placement FAILED! Parent:%s Child:%s Valid:%s",
             parent.name(),
             childName.c_str(),
             yes_no(child.isValid()));
  }
}

/// Converter for <PartSelector/> tags
template <>
void Converter<PartSelector>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  cms::DDParsingContext* const context = ns.context();
  dd4hep::SpecParRegistry& registry = *context->description.extension<dd4hep::SpecParRegistry>();
  xml_dim_t e(element);
  xml_dim_t specPar = e.parent();
  string specParName = specPar.attr<string>(_U(name));
  string path = e.attr<string>(DD_CMU(path));

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_specpars ? ALWAYS : DEBUG,
           "DD4CMS",
           "+++ PartSelector for %s path: %s",
           specParName.c_str(),
           path.c_str());

#endif

  size_t pos = std::string::npos;
  if ((pos = path.find("//.*:")) != std::string::npos) {
    path.erase(pos + 2, 3);
  }
  registry.specpars[specParName].paths.emplace_back(std::move(path));
}

/// Converter for <Parameter/> tags
template <>
void Converter<Parameter>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  cms::DDParsingContext* const context = ns.context();
  dd4hep::SpecParRegistry& registry = *context->description.extension<dd4hep::SpecParRegistry>();
  xml_dim_t e(element);
  xml_dim_t specPar = e.parent();
  xml_dim_t specParSect = specPar.parent();
  string specParName = specPar.attr<string>(_U(name));
  string name = e.nameStr();
  string value = e.attr<string>(DD_CMU(value));
  bool eval = e.hasAttr(_U(eval)) ? e.attr<bool>(_U(eval))
                                  : (specParSect.hasAttr(_U(eval)) ? specParSect.attr<bool>(_U(eval)) : false);
  string type = eval ? "number" : "string";

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_specpars ? ALWAYS : DEBUG,
           "DD4CMS",
           "+++ Parameter for %s: %s value %s is a %s",
           specParName.c_str(),
           name.c_str(),
           value.c_str(),
           type.c_str());

#endif

  size_t idx = value.find('[');
  if (idx == string::npos || type == "string") {
    registry.specpars[specParName].spars[name].emplace_back(std::move(value));
    return;
  }

  while (idx != string::npos) {
    ++idx;
    size_t idp = value.find(':', idx);
    size_t idq = value.find(']', idx);
    if (idp == string::npos || idp > idq)
      value.insert(idx, ns.name());
    else if (idp != string::npos && idp < idq)
      value[idp] = NAMESPACE_SEP;
    idx = value.find('[', idx);
  }

  string rep;
  string& v = value;
  size_t idq;
  for (idx = v.find('[', 0); idx != string::npos; idx = v.find('[', idx + 1)) {
    idq = v.find(']', idx + 1);
    rep = v.substr(idx + 1, idq - idx - 1);
    auto r = ns.context()->description.constants().find(rep);
    if (r != ns.context()->description.constants().end()) {
      rep = "(" + r->second->type + ")";
      v.replace(idx, idq - idx + 1, rep);
    }
  }
  registry.specpars[specParName].numpars[name].emplace_back(dd4hep::_toDouble(value));
}

template <typename TYPE>
static void convert_boolean(cms::DDParsingContext* context, xml_h element) {
  cms::DDNamespace ns(context);
  xml_dim_t e(element);
  string nam = e.nameStr();
  string solidName[2];
  Solid solids[2];
  Solid boolean;
  int cnt = 0;
  if (e.hasChild(DD_CMU(rSolid))) {
    for (xml_coll_t c(element, DD_CMU(rSolid)); cnt < 2 && c; ++c, ++cnt) {
      solidName[cnt] = c.attr<string>(_U(name));
      solids[cnt] = ns.solid(c.attr<string>(_U(name)));
    }
  } else {
    solidName[0] = e.attr<string>(DD_CMU(firstSolid));
    if ((solids[0] = ns.solid(e.attr<string>(DD_CMU(firstSolid)))).isValid())
      ++cnt;
    solidName[1] = e.attr<string>(DD_CMU(secondSolid));
    if ((solids[1] = ns.solid(e.attr<string>(DD_CMU(secondSolid)))).isValid())
      ++cnt;
  }
  if (cnt != 2) {
    except("DD4CMS", "+++ Failed to create boolean solid %s. Found only %d parts.", nam.c_str(), cnt);
  }

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_placements ? ALWAYS : DEBUG,
           "DD4CMS",
           "+++ BooleanSolid: %s Left: %-32s Right: %-32s",
           nam.c_str(),
           ((solids[0].ptr() == nullptr) ? solidName[0].c_str() : solids[0]->GetName()),
           ((solids[1].ptr() == nullptr) ? solidName[1].c_str() : solids[1]->GetName()));

#endif

  if (solids[0].isValid() && solids[1].isValid()) {
    Transform3D trafo;
    Converter<DDLTransform3D>(context->description, context, &trafo)(element);
    boolean = TYPE(solids[0], solids[1], trafo);
  } else {
    // Register it for later processing
    Transform3D trafo;
    Converter<DDLTransform3D>(context->description, context, &trafo)(element);
    ns.context()->unresolvedShapes.emplace(nam,
                                           DDParsingContext::BooleanShape<TYPE>(solidName[0], solidName[1], trafo));
  }
  if (!boolean.isValid()) {
    // Delay processing the shape
    ns.context()->shapes.emplace(nam, dd4hep::Solid(nullptr));
  } else
    ns.addSolid(nam, boolean);
}

/// Converter for <UnionSolid/> tags
template <>
void Converter<DDLUnionSolid>::operator()(xml_h element) const {
  convert_boolean<UnionSolid>(_param<cms::DDParsingContext>(), element);
}

/// Converter for <SubtractionSolid/> tags
template <>
void Converter<DDLSubtractionSolid>::operator()(xml_h element) const {
  convert_boolean<SubtractionSolid>(_param<cms::DDParsingContext>(), element);
}

/// Converter for <IntersectionSolid/> tags
template <>
void Converter<DDLIntersectionSolid>::operator()(xml_h element) const {
  convert_boolean<IntersectionSolid>(_param<cms::DDParsingContext>(), element);
}

/// Converter for <Polycone/> tags
template <>
void Converter<DDLPolycone>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam = e.nameStr();
  double startPhi = ns.attr<double>(e, DD_CMU(startPhi));
  double deltaPhi = ns.attr<double>(e, DD_CMU(deltaPhi));
  vector<double> z, rmin, rmax, r;

  for (xml_coll_t rzpoint(element, DD_CMU(RZPoint)); rzpoint; ++rzpoint) {
    z.emplace_back(ns.attr<double>(rzpoint, _U(z)));
    r.emplace_back(ns.attr<double>(rzpoint, _U(r)));
  }
  if (z.empty()) {
    for (xml_coll_t zplane(element, DD_CMU(ZSection)); zplane; ++zplane) {
      rmin.emplace_back(ns.attr<double>(zplane, DD_CMU(rMin)));
      rmax.emplace_back(ns.attr<double>(zplane, DD_CMU(rMax)));
      z.emplace_back(ns.attr<double>(zplane, _U(z)));
    }

#ifdef EDM_ML_DEBUG

    printout(ns.context()->debug_shapes ? ALWAYS : DEBUG,
             "DD4CMS",
             "+   Polycone: startPhi=%10.3f [rad] deltaPhi=%10.3f [rad]  %4ld z-planes",
             startPhi,
             deltaPhi,
             z.size());

#endif

    ns.addSolid(nam, Polycone(startPhi, deltaPhi, rmin, rmax, z));
  } else {
#ifdef EDM_ML_DEBUG

    printout(ns.context()->debug_shapes ? ALWAYS : DEBUG,
             "DD4CMS",
             "+   Polycone: startPhi=%10.3f [rad] deltaPhi=%10.3f [rad]  %4ld z-planes and %4ld radii",
             startPhi,
             deltaPhi,
             z.size(),
             r.size());

#endif

    ns.addSolid(nam, Polycone(startPhi, deltaPhi, r, z));
  }
}

/// Converter for <ExtrudedPolygon/> tags
template <>
void Converter<DDLExtrudedPolygon>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam = e.nameStr();
  vector<double> pt_x, pt_y, sec_x, sec_y, sec_z, sec_scale;

  for (xml_coll_t sec(element, DD_CMU(ZXYSection)); sec; ++sec) {
    sec_z.emplace_back(ns.attr<double>(sec, _U(z)));
    sec_x.emplace_back(ns.attr<double>(sec, _U(x)));
    sec_y.emplace_back(ns.attr<double>(sec, _U(y)));
    sec_scale.emplace_back(ns.attr<double>(sec, DD_CMU(scale), 1.0));
  }
  for (xml_coll_t pt(element, DD_CMU(XYPoint)); pt; ++pt) {
    pt_x.emplace_back(ns.attr<double>(pt, _U(x)));
    pt_y.emplace_back(ns.attr<double>(pt, _U(y)));
  }

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_shapes ? ALWAYS : DEBUG,
           "DD4CMS",
           "+   ExtrudedPolygon: %4ld points %4ld zxy sections",
           pt_x.size(),
           sec_z.size());

#endif

  ns.addSolid(nam, ExtrudedPolygon(pt_x, pt_y, sec_z, sec_x, sec_y, sec_scale));
}

/// Converter for <Polyhedra/> tags
template <>
void Converter<DDLPolyhedra>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam = e.nameStr();
  double numSide = ns.attr<int>(e, DD_CMU(numSide));
  double startPhi = ns.attr<double>(e, DD_CMU(startPhi));
  double deltaPhi = ns.attr<double>(e, DD_CMU(deltaPhi));
  vector<double> z, rmin, rmax;

  for (xml_coll_t zplane(element, DD_CMU(RZPoint)); zplane; ++zplane) {
    rmin.emplace_back(0.0);
    rmax.emplace_back(ns.attr<double>(zplane, _U(r)));
    z.emplace_back(ns.attr<double>(zplane, _U(z)));
  }
  for (xml_coll_t zplane(element, DD_CMU(ZSection)); zplane; ++zplane) {
    rmin.emplace_back(ns.attr<double>(zplane, DD_CMU(rMin)));
    rmax.emplace_back(ns.attr<double>(zplane, DD_CMU(rMax)));
    z.emplace_back(ns.attr<double>(zplane, _U(z)));
  }

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_shapes ? ALWAYS : DEBUG,
           "DD4CMS",
           "+   Polyhedra:startPhi=%8.3f [rad] deltaPhi=%8.3f [rad]  %4d sides %4ld z-planes",
           startPhi,
           deltaPhi,
           numSide,
           z.size());

#endif

  ns.addSolid(nam, Polyhedra(numSide, startPhi, deltaPhi, z, rmin, rmax));
}

/// Converter for <Sphere/> tags
template <>
void Converter<DDLSphere>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam = e.nameStr();
  double rinner = ns.attr<double>(e, DD_CMU(innerRadius));
  double router = ns.attr<double>(e, DD_CMU(outerRadius));
  double startPhi = ns.attr<double>(e, DD_CMU(startPhi));
  double deltaPhi = ns.attr<double>(e, DD_CMU(deltaPhi));
  double startTheta = ns.attr<double>(e, DD_CMU(startTheta));
  double deltaTheta = ns.attr<double>(e, DD_CMU(deltaTheta));

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_shapes ? ALWAYS : DEBUG,
           "DD4CMS",
           "+   Sphere:   r_inner=%8.3f [cm] r_outer=%8.3f [cm]"
           " startPhi=%8.3f [rad] deltaPhi=%8.3f startTheta=%8.3f delteTheta=%8.3f [rad]",
           rinner,
           router,
           startPhi,
           deltaPhi,
           startTheta,
           deltaTheta);

#endif

  ns.addSolid(nam, Sphere(rinner, router, startTheta, deltaTheta, startPhi, deltaPhi));
}

/// Converter for <Torus/> tags
template <>
void Converter<DDLTorus>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam = e.nameStr();
  double r = ns.attr<double>(e, DD_CMU(torusRadius));
  double rinner = ns.attr<double>(e, DD_CMU(innerRadius));
  double router = ns.attr<double>(e, DD_CMU(outerRadius));
  double startPhi = ns.attr<double>(e, DD_CMU(startPhi));
  double deltaPhi = ns.attr<double>(e, DD_CMU(deltaPhi));

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_shapes ? ALWAYS : DEBUG,
           "DD4CMS",
           "+   Torus:    r=%10.3f [cm] r_inner=%10.3f [cm] r_outer=%10.3f [cm]"
           " startPhi=%10.3f [rad] deltaPhi=%10.3f [rad]",
           r,
           rinner,
           router,
           startPhi,
           deltaPhi);

#endif

  ns.addSolid(nam, Torus(r, rinner, router, startPhi, deltaPhi));
}

/// Converter for <Pseudotrap/> tags
template <>
void Converter<DDLPseudoTrap>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam = e.nameStr();
  double dx1 = ns.attr<double>(e, DD_CMU(dx1));
  double dy1 = ns.attr<double>(e, DD_CMU(dy1));
  double dx2 = ns.attr<double>(e, DD_CMU(dx2));
  double dy2 = ns.attr<double>(e, DD_CMU(dy2));
  double dz = ns.attr<double>(e, _U(dz));
  double r = ns.attr<double>(e, _U(radius));
  bool atMinusZ = ns.attr<bool>(e, DD_CMU(atMinusZ));

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_shapes ? ALWAYS : DEBUG,
           "DD4CMS",
           "+   Pseudotrap:  dz=%8.3f [cm] dx1:%.3f dy1:%.3f dx2=%.3f dy2=%.3f radius:%.3f atMinusZ:%s",
           dz,
           dx1,
           dy1,
           dx2,
           dy2,
           r,
           yes_no(atMinusZ));

#endif

  ns.addSolid(nam, PseudoTrap(dx1, dx2, dy1, dy2, dz, r, atMinusZ));
}

/// Converter for <Trapezoid/> tags
template <>
void Converter<DDLTrapezoid>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam = e.nameStr();
  double dz = ns.attr<double>(e, _U(dz));
  double alp1 = ns.attr<double>(e, DD_CMU(alp1));
  double bl1 = ns.attr<double>(e, DD_CMU(bl1));
  double tl1 = ns.attr<double>(e, DD_CMU(tl1));
  double h1 = ns.attr<double>(e, DD_CMU(h1));
  double alp2 = ns.attr<double>(e, DD_CMU(alp2));
  double bl2 = ns.attr<double>(e, DD_CMU(bl2));
  double tl2 = ns.attr<double>(e, DD_CMU(tl2));
  double h2 = ns.attr<double>(e, DD_CMU(h2));
  double phi = ns.attr<double>(e, _U(phi), 0.0);
  double theta = ns.attr<double>(e, _U(theta), 0.0);

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_shapes ? ALWAYS : DEBUG,
           "DD4CMS",
           "+   Trapezoid:  dz=%10.3f [cm] alp1:%.3f bl1=%.3f tl1=%.3f alp2=%.3f bl2=%.3f tl2=%.3f h2=%.3f phi=%.3f "
           "theta=%.3f",
           dz,
           alp1,
           bl1,
           tl1,
           h1,
           alp2,
           bl2,
           tl2,
           h2,
           phi,
           theta);

#endif

  ns.addSolid(nam, Trap(dz, theta, phi, h1, bl1, tl1, alp1, h2, bl2, tl2, alp2));
}

/// Converter for <Trd1/> tags
template <>
void Converter<DDLTrd1>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam = e.nameStr();
  double dx1 = ns.attr<double>(e, DD_CMU(dx1));
  double dy1 = ns.attr<double>(e, DD_CMU(dy1));
  double dx2 = ns.attr<double>(e, DD_CMU(dx2), 0.0);
  double dy2 = ns.attr<double>(e, DD_CMU(dy2), dy1);
  double dz = ns.attr<double>(e, DD_CMU(dz));
  if (dy1 == dy2) {
#ifdef EDM_ML_DEBUG

    printout(ns.context()->debug_shapes ? ALWAYS : DEBUG,
             "DD4CMS",
             "+   Trd1:       dz=%8.3f [cm] dx1:%.3f dy1:%.3f dx2:%.3f dy2:%.3f",
             dz,
             dx1,
             dy1,
             dx2,
             dy2);

#endif

    ns.addSolid(nam, Trd1(dx1, dx2, dy1, dz));
  } else {
#ifdef EDM_ML_DEBUG

    printout(ns.context()->debug_shapes ? ALWAYS : DEBUG,
             "DD4CMS",
             "+   Trd1(which is actually Trd2):       dz=%8.3f [cm] dx1:%.3f dy1:%.3f dx2:%.3f dy2:%.3f",
             dz,
             dx1,
             dy1,
             dx2,
             dy2);

#endif

    ns.addSolid(nam, Trd2(dx1, dx2, dy1, dy2, dz));
  }
}

/// Converter for <Trd2/> tags
template <>
void Converter<DDLTrd2>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam = e.nameStr();
  double dx1 = ns.attr<double>(e, DD_CMU(dx1));
  double dy1 = ns.attr<double>(e, DD_CMU(dy1));
  double dx2 = ns.attr<double>(e, DD_CMU(dx2), 0.0);
  double dy2 = ns.attr<double>(e, DD_CMU(dy2), dy1);
  double dz = ns.attr<double>(e, DD_CMU(dz));

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_shapes ? ALWAYS : DEBUG,
           "DD4CMS",
           "+   Trd1:       dz=%8.3f [cm] dx1:%.3f dy1:%.3f dx2:%.3f dy2:%.3f",
           dz,
           dx1,
           dy1,
           dx2,
           dy2);

#endif

  ns.addSolid(nam, Trd2(dx1, dx2, dy1, dy2, dz));
}

/// Converter for <Tubs/> tags
template <>
void Converter<DDLTubs>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam = e.nameStr();
  double dz = ns.attr<double>(e, DD_CMU(dz));
  double rmin = ns.attr<double>(e, DD_CMU(rMin));
  double rmax = ns.attr<double>(e, DD_CMU(rMax));
  double startPhi = ns.attr<double>(e, DD_CMU(startPhi), 0.0);
  double deltaPhi = ns.attr<double>(e, DD_CMU(deltaPhi), 2 * M_PI);

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_shapes ? ALWAYS : DEBUG,
           "DD4CMS",
           "+   Tubs:     dz=%8.3f [cm] rmin=%8.3f [cm] rmax=%8.3f [cm]"
           " startPhi=%8.3f [rad] deltaPhi=%8.3f [rad]",
           dz,
           rmin,
           rmax,
           startPhi,
           deltaPhi);

#endif

  ns.addSolid(nam, Tube(rmin, rmax, dz, startPhi, startPhi + deltaPhi));
}

/// Converter for <CutTubs/> tags
template <>
void Converter<DDLCutTubs>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam = e.nameStr();
  double dz = ns.attr<double>(e, DD_CMU(dz));
  double rmin = ns.attr<double>(e, DD_CMU(rMin));
  double rmax = ns.attr<double>(e, DD_CMU(rMax));
  double startPhi = ns.attr<double>(e, DD_CMU(startPhi));
  double deltaPhi = ns.attr<double>(e, DD_CMU(deltaPhi));
  double lx = ns.attr<double>(e, DD_CMU(lx));
  double ly = ns.attr<double>(e, DD_CMU(ly));
  double lz = ns.attr<double>(e, DD_CMU(lz));
  double tx = ns.attr<double>(e, DD_CMU(tx));
  double ty = ns.attr<double>(e, DD_CMU(ty));
  double tz = ns.attr<double>(e, DD_CMU(tz));

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_shapes ? ALWAYS : DEBUG,
           "DD4CMS",
           "+   CutTube:  dz=%8.3f [cm] rmin=%8.3f [cm] rmax=%8.3f [cm]"
           " startPhi=%8.3f [rad] deltaPhi=%8.3f [rad]...",
           dz,
           rmin,
           rmax,
           startPhi,
           deltaPhi);

#endif

  ns.addSolid(nam, CutTube(rmin, rmax, dz, startPhi, startPhi + deltaPhi, lx, ly, lz, tx, ty, tz));
}

/// Converter for <TruncTubs/> tags
template <>
void Converter<DDLTruncTubs>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam = e.nameStr();
  double zhalf = ns.attr<double>(e, DD_CMU(zHalf));
  double rmin = ns.attr<double>(e, DD_CMU(rMin));
  double rmax = ns.attr<double>(e, DD_CMU(rMax));
  double startPhi = ns.attr<double>(e, DD_CMU(startPhi));
  double deltaPhi = ns.attr<double>(e, DD_CMU(deltaPhi));
  double cutAtStart = ns.attr<double>(e, DD_CMU(cutAtStart));
  double cutAtDelta = ns.attr<double>(e, DD_CMU(cutAtDelta));
  bool cutInside = ns.attr<bool>(e, DD_CMU(cutInside));

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_shapes ? ALWAYS : DEBUG,
           "DD4CMS",
           "+   TruncTube:zHalf=%8.3f [cm] rmin=%8.3f [cm] rmax=%8.3f [cm]"
           " startPhi=%8.3f [rad] deltaPhi=%8.3f [rad] atStart=%8.3f [cm] atDelta=%8.3f [cm] inside:%s",
           zhalf,
           rmin,
           rmax,
           startPhi,
           deltaPhi,
           cutAtStart,
           cutAtDelta,
           yes_no(cutInside));

#endif

  ns.addSolid(nam, TruncatedTube(zhalf, rmin, rmax, startPhi, deltaPhi, cutAtStart, cutAtDelta, cutInside));
}

/// Converter for <EllipticalTube/> tags
template <>
void Converter<DDLEllipticalTube>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam = e.nameStr();
  double dx = ns.attr<double>(e, DD_CMU(xSemiAxis));
  double dy = ns.attr<double>(e, DD_CMU(ySemiAxis));
  double dz = ns.attr<double>(e, DD_CMU(zHeight));

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_shapes ? ALWAYS : DEBUG,
           "DD4CMS",
           "+   EllipticalTube xSemiAxis=%8.3f [cm] ySemiAxis=%8.3f [cm] zHeight=%8.3f [cm]",
           dx,
           dy,
           dz);

#endif

  ns.addSolid(nam, EllipticalTube(dx, dy, dz));
}

/// Converter for <Cone/> tags
template <>
void Converter<DDLCone>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam = e.nameStr();
  double dz = ns.attr<double>(e, DD_CMU(dz));
  double rmin1 = ns.attr<double>(e, DD_CMU(rMin1));
  double rmin2 = ns.attr<double>(e, DD_CMU(rMin2));
  double rmax1 = ns.attr<double>(e, DD_CMU(rMax1));
  double rmax2 = ns.attr<double>(e, DD_CMU(rMax2));
  double startPhi = ns.attr<double>(e, DD_CMU(startPhi));
  double deltaPhi = ns.attr<double>(e, DD_CMU(deltaPhi));
  double phi2 = startPhi + deltaPhi;

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_shapes ? ALWAYS : DEBUG,
           "DD4CMS",
           "+   Cone:     dz=%8.3f [cm]"
           " rmin1=%8.3f [cm] rmax1=%8.3f [cm]"
           " rmin2=%8.3f [cm] rmax2=%8.3f [cm]"
           " startPhi=%8.3f [rad] deltaPhi=%8.3f [rad]",
           dz,
           rmin1,
           rmax1,
           rmin2,
           rmax2,
           startPhi,
           deltaPhi);

#endif

  ns.addSolid(nam, ConeSegment(dz, rmin1, rmax1, rmin2, rmax2, startPhi, phi2));
}

/// Converter for <Shapeless/> tags
template <>
void Converter<DDLShapeless>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam = e.nameStr();

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_shapes ? ALWAYS : DEBUG,
           "DD4CMS",
           "+   Shapeless: THIS ONE CAN ONLY BE USED AT THE VOLUME LEVEL -> Assembly%s",
           nam.c_str());

#endif

  ns.addSolid(nam, Box(1, 1, 1));
}

/// Converter for <Box/> tags
template <>
void Converter<DDLBox>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam = e.nameStr();
  double dx = ns.attr<double>(e, DD_CMU(dx));
  double dy = ns.attr<double>(e, DD_CMU(dy));
  double dz = ns.attr<double>(e, DD_CMU(dz));

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_shapes ? ALWAYS : DEBUG,
           "DD4CMS",
           "+   Box:      dx=%10.3f [cm] dy=%10.3f [cm] dz=%10.3f [cm]",
           dx,
           dy,
           dz);

#endif

  ns.addSolid(nam, Box(dx, dy, dz));
}

/// DD4hep specific Converter for <Include/> tags: process only the constants
template <>
void Converter<include_load>::operator()(xml_h element) const {
  string fname = element.attr<string>(_U(ref));
  edm::FileInPath fp(fname);
  xml::Document doc;
  doc = xml::DocumentHandler().load(fp.fullPath());

#ifdef EDM_ML_DEBUG

  printout(_param<cms::DDParsingContext>()->debug_includes ? ALWAYS : DEBUG,
           "DD4CMS",
           "+++ Processing the CMS detector description %s",
           fname.c_str());

#endif

  _option<DDRegistry>()->includes.emplace_back(doc);
}

/// DD4hep specific Converter for <Include/> tags: process only the constants
template <>
void Converter<include_unload>::operator()(xml_h element) const {
  string fname = xml::DocumentHandler::system_path(element);
  xml::DocumentHolder(xml_elt_t(element).document()).assign(nullptr);

#ifdef EDM_ML_DEBUG

  printout(_param<cms::DDParsingContext>()->debug_includes ? ALWAYS : DEBUG,
           "DD4CMS",
           "+++ Finished processing %s",
           fname.c_str());
#endif
}

/// DD4hep specific Converter for <Include/> tags: process only the constants
template <>
void Converter<include_constants>::operator()(xml_h element) const {
  xml_coll_t(element, DD_CMU(ConstantsSection)).for_each(Converter<ConstantsSection>(description, param, optional));
}

namespace {

  //  The meaning of the axis index is the following:
  //    for all volumes having shapes like box, trd1, trd2, trap, gtra or para - 1,2,3 means X,Y,Z;
  //    for tube, tubs, cone, cons - 1 means Rxy, 2 means phi and 3 means Z;
  //    for pcon and pgon - 2 means phi and 3 means Z;
  //    for spheres 1 means R and 2 means phi.

  enum class DDAxes { x = 1, y = 2, z = 3, rho = 1, phi = 2, undefined };
  const std::map<std::string, DDAxes> axesmap{{"x", DDAxes::x},
                                              {"y", DDAxes::y},
                                              {"z", DDAxes::z},
                                              {"rho", DDAxes::rho},
                                              {"phi", DDAxes::phi},
                                              {"undefined", DDAxes::undefined}};
}  // namespace

/// Converter for <Division/> tags
template <>
void Converter<DDLDivision>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>(), element);
  xml_dim_t e(element);
  string childName = e.nameStr();
  if (strchr(childName.c_str(), NAMESPACE_SEP) == nullptr)
    childName = ns.prepend(childName);

  string parentName = ns.attr<string>(e, DD_CMU(parent));
  if (strchr(parentName.c_str(), NAMESPACE_SEP) == nullptr)
    parentName = ns.prepend(parentName);
  string axis = ns.attr<string>(e, DD_CMU(axis));

  // If you divide a tube of 360 degrees the offset displaces
  // the starting angle, but you still fill the 360 degrees
  double offset = e.hasAttr(DD_CMU(offset)) ? ns.attr<double>(e, DD_CMU(offset)) : 0e0;
  double width = e.hasAttr(DD_CMU(width)) ? ns.attr<double>(e, DD_CMU(width)) : 0e0;
  int nReplicas = e.hasAttr(DD_CMU(nReplicas)) ? ns.attr<int>(e, DD_CMU(nReplicas)) : 0;

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_placements ? ALWAYS : DEBUG,
           "DD4CMS",
           "+++ Start executing Division of %s along %s (%d) with offset %6.3f and %6.3f to produce %s....",
           parentName.c_str(),
           axis.c_str(),
           axesmap.at(axis),
           offset,
           width,
           childName.c_str());

#endif

  Volume parent = ns.volume(parentName);

  const TGeoShape* shape = parent.solid();
  TClass* cl = shape->IsA();
  if (cl == TGeoTubeSeg::Class()) {
    const TGeoTubeSeg* sh = (const TGeoTubeSeg*)shape;
    double widthInDeg = convertRadToDeg(width);
    double startInDeg = convertRadToDeg(offset);
    int numCopies = (int)((sh->GetPhi2() - sh->GetPhi1()) / widthInDeg);

#ifdef EDM_ML_DEBUG

    printout(ns.context()->debug_placements ? ALWAYS : DEBUG,
             "DD4CMS",
             "+++    ...divide %s along %s (%d) with offset %6.3f deg and %6.3f deg to produce %d copies",
             parent.solid().type(),
             axis.c_str(),
             axesmap.at(axis),
             startInDeg,
             widthInDeg,
             numCopies);

#endif

    Volume child = parent.divide(childName, static_cast<int>(axesmap.at(axis)), numCopies, startInDeg, widthInDeg);

    ns.context()->volumes[childName] = child;

#ifdef EDM_ML_DEBUG

    printout(ns.context()->debug_placements ? ALWAYS : DEBUG,
             "DD4CMS",
             "+++ %s Parent: %-24s [%s] Child: %-32s [%s] is multivolume [%s]",
             e.tag().c_str(),
             parentName.c_str(),
             parent.isValid() ? "VALID" : "INVALID",
             child.name(),
             child.isValid() ? "VALID" : "INVALID",
             child->IsVolumeMulti() ? "YES" : "NO");
#endif

  } else if (cl == TGeoTrd1::Class()) {
    double dy = static_cast<const TGeoTrd1*>(shape)->GetDy();

#ifdef EDM_ML_DEBUG

    printout(ns.context()->debug_placements ? ALWAYS : DEBUG,
             "DD4CMS",
             "+++    ...divide %s along %s (%d) with offset %6.3f cm and %6.3f cm to produce %d copies in %6.3f",
             parent.solid().type(),
             axis.c_str(),
             axesmap.at(axis),
             -dy + offset + width,
             width,
             nReplicas,
             dy);

#endif

    Volume child = parent.divide(childName, static_cast<int>(axesmap.at(axis)), nReplicas, -dy + offset + width, width);

    ns.context()->volumes[childName] = child;

#ifdef EDM_ML_DEBUG

    printout(ns.context()->debug_placements ? ALWAYS : DEBUG,
             "DD4CMS",
             "+++ %s Parent: %-24s [%s] Child: %-32s [%s] is multivolume [%s]",
             e.tag().c_str(),
             parentName.c_str(),
             parent.isValid() ? "VALID" : "INVALID",
             child.name(),
             child.isValid() ? "VALID" : "INVALID",
             child->IsVolumeMulti() ? "YES" : "NO");

#endif

  } else {
    printout(ERROR, "DD4CMS", "++ FAILED Division of a %s is not implemented yet!", parent.solid().type());
  }
}

/// Converter for <Algorithm/> tags
template <>
void Converter<DDLAlgorithm>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string name = e.nameStr();
  size_t idx;
  string type = "DDCMS_" + ns.realName(name);
  while ((idx = type.find(NAMESPACE_SEP)) != string::npos)
    type[idx] = '_';

#ifdef EDM_ML_DEBUG

  printout(
      ns.context()->debug_algorithms ? ALWAYS : DEBUG, "DD4CMS", "+++ Start executing algorithm %s....", type.c_str());

#endif

  long ret = PluginService::Create<long>(type, &description, ns.context(), &element);
  if (ret == s_executed) {
#ifdef EDM_ML_DEBUG

    printout(ns.context()->debug_algorithms ? ALWAYS : DEBUG,

             "DD4CMS",
             "+++ Executed algorithm: %08lX = %s",
             ret,
             name.c_str());

#endif
    return;
  }
  printout(ERROR, "DD4CMS", "++ FAILED  NOT ADDING SUBDETECTOR %08lX = %s", ret, name.c_str());
}

template <class InputIt, class ForwardIt, class BinOp>
void for_each_token(InputIt first, InputIt last, ForwardIt s_first, ForwardIt s_last, BinOp binary_op) {
  while (first != last) {
    const auto pos = std::find_first_of(first, last, s_first, s_last);
    binary_op(first, pos);
    if (pos == last)
      break;
    first = std::next(pos);
  }
}

namespace {

  std::vector<string> splitString(const string& str, const string& delims = ",") {
    std::vector<string> output;

    for_each_token(cbegin(str), cend(str), cbegin(delims), cend(delims), [&output](auto first, auto second) {
      if (first != second) {
        if (string(first, second).front() == '[' && string(first, second).back() == ']') {
          first++;
          second--;
        }
        output.emplace_back(string(first, second));
      }
    });
    return output;
  }

  std::vector<double> splitNumeric(const string& str, const string& delims = ",") {
    std::vector<double> output;

    for_each_token(cbegin(str), cend(str), cbegin(delims), cend(delims), [&output](auto first, auto second) {
      if (first != second) {
        if (string(first, second).front() == '[' && string(first, second).back() == ']') {
          first++;
          second--;
        }
        output.emplace_back(dd4hep::_toDouble(string(first, second)));
      }
    });
    return output;
  }
}  // namespace

/// Converter for <Vector/> tags
/// FIXME: Check if(parent() == "Algorithm" || parent() == "SpecPar")
template <>
void Converter<DDLVector>::operator()(xml_h element) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  cms::DDParsingContext* const context = ns.context();
  DDVectorsMap* registry = context->description.extension<DDVectorsMap>();
  xml_dim_t e(element);
  string name = ns.prepend(e.nameStr());
  string type = ns.attr<string>(e, _U(type));
  string nEntries = ns.attr<string>(e, DD_CMU(nEntries));
  string val = e.text();
  val.erase(remove_if(val.begin(), val.end(), [](unsigned char x) { return isspace(x); }), val.end());

#ifdef EDM_ML_DEBUG

  printout(ns.context()->debug_constants ? ALWAYS : DEBUG,
           "DD4CMS",
           "+++ Vector<%s>:  %s[%s]: %s",
           type.c_str(),
           name.c_str(),
           nEntries.c_str(),
           val.c_str());

#endif

  try {
    std::vector<double> results = splitNumeric(val);
    registry->insert(
        {name,
         results});  //tbb::concurrent_vector<double, tbb::cache_aligned_allocator<double>>(results.begin(), results.end())});
  } catch (const exception& e) {
#ifdef EDM_ML_DEBUG

    printout(INFO,
             "DD4CMS",
             "++ Unresolved Vector<%s>:  %s[%s]: %s. Try to resolve later. [%s]",
             type.c_str(),
             name.c_str(),
             nEntries.c_str(),
             val.c_str(),
             e.what());

#endif

    std::vector<string> results = splitString(val);
    context->unresolvedVectors.insert({name, results});
  }
}

template <>
void Converter<debug>::operator()(xml_h dbg) const {
  cms::DDNamespace ns(_param<cms::DDParsingContext>());
  if (dbg.hasChild(DD_CMU(debug_constants)))
    ns.setContext()->debug_constants = true;
  if (dbg.hasChild(DD_CMU(debug_materials)))
    ns.setContext()->debug_materials = true;
  if (dbg.hasChild(DD_CMU(debug_rotations)))
    ns.setContext()->debug_rotations = true;
  if (dbg.hasChild(DD_CMU(debug_shapes)))
    ns.setContext()->debug_shapes = true;
  if (dbg.hasChild(DD_CMU(debug_volumes)))
    ns.setContext()->debug_volumes = true;
  if (dbg.hasChild(DD_CMU(debug_placements)))
    ns.setContext()->debug_placements = true;
  if (dbg.hasChild(DD_CMU(debug_namespaces)))
    ns.setContext()->debug_namespaces = true;
  if (dbg.hasChild(DD_CMU(debug_includes)))
    ns.setContext()->debug_includes = true;
  if (dbg.hasChild(DD_CMU(debug_algorithms)))
    ns.setContext()->debug_algorithms = true;
  if (dbg.hasChild(DD_CMU(debug_specpars)))
    ns.setContext()->debug_specpars = true;
}

template <>
void Converter<DDRegistry>::operator()(xml_h /* element */) const {
  cms::DDParsingContext* context = _param<cms::DDParsingContext>();
  DDRegistry* res = _option<DDRegistry>();
  cms::DDNamespace ns(context);
  int count = 0;

#ifdef EDM_ML_DEBUG

  printout(context->debug_constants ? ALWAYS : DEBUG,
           "DD4CMS",
           "+++ RESOLVING %ld unknown constants..... (out of %ld)",
           res->unresolvedConst.size(),
           res->originalConst.size());
#endif

  while (!res->unresolvedConst.empty()) {
    for (auto& i : res->unresolvedConst) {
      const string& n = i.first;
      string rep;
      string& v = i.second;
      size_t idx, idq;
      for (idx = v.find('[', 0); idx != string::npos; idx = v.find('[', idx + 1)) {
        idq = v.find(']', idx + 1);
        rep = v.substr(idx + 1, idq - idx - 1);
        auto r = res->originalConst.find(rep);
        if (r != res->originalConst.end()) {
          rep = "(" + (*r).second + ")";
          v.replace(idx, idq - idx + 1, rep);
        }
      }
      if (v.find(']') == string::npos) {
        if (v.find("-+") != string::npos || v.find("+-") != string::npos) {
          while ((idx = v.find("-+")) != string::npos)
            v.replace(idx, 2, "-");
          while ((idx = v.find("+-")) != string::npos)
            v.replace(idx, 2, "-");
        }

#ifdef EDM_ML_DEBUG

        printout(context->debug_constants ? ALWAYS : DEBUG,
                 "DD4CMS",
                 "+++ [%06ld] ----------  %-40s = %s",
                 res->unresolvedConst.size() - 1,
                 n.c_str(),
                 res->originalConst[n].c_str());

#endif

        ns.addConstantNS(n, v, "number");
        res->unresolvedConst.erase(n);
        break;
      }
    }
    if (++count > 10000)
      break;
  }
  if (!res->unresolvedConst.empty()) {
    for (const auto& e : res->unresolvedConst)
      printout(ERROR, "DD4CMS", "+++ Unresolved constant: %-40s = %s.", e.first.c_str(), e.second.c_str());
    except("DD4CMS", "++ FAILED to resolve %ld constant entries:", res->unresolvedConst.size());
  }
  res->unresolvedConst.clear();
  res->originalConst.clear();
}

template <>
void Converter<print_xml_doc>::operator()(xml_h element) const {
  string fname = xml::DocumentHandler::system_path(element);

#ifdef EDM_ML_DEBUG

  printout(_param<cms::DDParsingContext>()->debug_includes ? ALWAYS : DEBUG,
           "DD4CMS",
           "+++ Processing data from: %s",
           fname.c_str());

#endif
}

/// Converter for <DDDefinition/> tags
static long load_dddefinition(Detector& det, xml_h element) {
  xml_elt_t dddef(element);
  if (dddef) {
    cms::DDParsingContext context(det);
    cms::DDNamespace ns(context);
    ns.addConstantNS("world_x", "101*m", "number");
    ns.addConstantNS("world_y", "101*m", "number");
    ns.addConstantNS("world_z", "450*m", "number");
    ns.addConstantNS("Air", "materials:Air", "string");
    ns.addConstantNS("Vacuum", "materials:Vacuum", "string");
    ns.addConstantNS("fm", "1e-12*m", "number");
    ns.addConstantNS("mum", "1e-6*m", "number");

    string fname = xml::DocumentHandler::system_path(element);
    bool open_geometry = dddef.hasChild(DD_CMU(open_geometry)) ? dddef.child(DD_CMU(open_geometry)) : true;
    bool close_geometry = dddef.hasChild(DD_CMU(close_geometry)) ? dddef.hasChild(DD_CMU(close_geometry)) : true;

    xml_coll_t(dddef, _U(debug)).for_each(Converter<debug>(det, &context));

    // Here we define the order how XML elements are processed.
    // Be aware of dependencies. This can only defined once.
    // At the end it is a limitation of DOM....
    printout(INFO, "DD4CMS", "+++ Processing the CMS detector description %s", fname.c_str());

    xml::Document doc;
    Converter<print_xml_doc> print_doc(det, &context);
    try {
      DDRegistry res;
      res.unresolvedConst.reserve(2000);
      res.originalConst.reserve(6000);
      print_doc((doc = dddef.document()).root());
      xml_coll_t(dddef, DD_CMU(ConstantsSection)).for_each(Converter<ConstantsSection>(det, &context, &res));
      xml_coll_t(dddef, DD_CMU(RotationSection)).for_each(Converter<RotationSection>(det, &context));
      xml_coll_t(dddef, DD_CMU(MaterialSection)).for_each(Converter<MaterialSection>(det, &context));

      xml_coll_t(dddef, DD_CMU(IncludeSection)).for_each(DD_CMU(Include), Converter<include_load>(det, &context, &res));

      for (xml::Document d : res.includes) {
        print_doc((doc = d).root());
        Converter<include_constants>(det, &context, &res)((doc = d).root());
      }
      // Before we continue, we have to resolve all constants NOW!
      Converter<DDRegistry>(det, &context, &res)(dddef);
      {
        DDVectorsMap* registry = context.description.extension<DDVectorsMap>();

        printout(context.debug_constants ? ALWAYS : DEBUG,
                 "DD4CMS",
                 "+++ RESOLVING %ld Vectors.....",
                 context.unresolvedVectors.size());

        while (!context.unresolvedVectors.empty()) {
          for (auto it = context.unresolvedVectors.begin(); it != context.unresolvedVectors.end();) {
            std::vector<double> result;
            for (const auto& i : it->second) {
              result.emplace_back(dd4hep::_toDouble(i));
            }
            registry->insert({it->first, result});
            // All components are resolved
            it = context.unresolvedVectors.erase(it);
          }
        }
      }
      // Now we can process the include files one by one.....
      for (xml::Document d : res.includes) {
        print_doc((doc = d).root());
        xml_coll_t(d.root(), DD_CMU(MaterialSection)).for_each(Converter<MaterialSection>(det, &context));
      }
      {
        printout(context.debug_materials ? ALWAYS : DEBUG,
                 "DD4CMS",
                 "+++ RESOLVING %ld unknown material constituents.....",
                 context.unresolvedMaterials.size());

        // Resolve referenced materials (if any)

        while (!context.unresolvedMaterials.empty()) {
          for (auto it = context.unresolvedMaterials.begin(); it != context.unresolvedMaterials.end();) {
            auto const& name = it->first;
            std::vector<bool> valid;

            printout(context.debug_materials ? ALWAYS : DEBUG,
                     "DD4CMS",
                     "+++ [%06ld] ----------  %s",
                     context.unresolvedMaterials.size(),
                     name.c_str());

            auto mat = ns.material(name);
            for (auto& mit : it->second) {
              printout(context.debug_materials ? ALWAYS : DEBUG,
                       "DD4CMS",
                       "+++           component  %-48s Fraction: %.6f",
                       mit.name.c_str(),
                       mit.fraction);
              auto fmat = ns.material(mit.name);
              if (nullptr != fmat.ptr()) {
                if (mat.ptr()->GetMaterial()->IsMixture()) {
                  valid.emplace_back(true);
                  static_cast<TGeoMixture*>(mat.ptr()->GetMaterial())
                      ->AddElement(fmat.ptr()->GetMaterial(), mit.fraction);
                }
              }
            }
            // All components are resolved
            if (valid.size() == it->second.size())
              it = context.unresolvedMaterials.erase(it);
            else
              ++it;
          }
          // Do it again if there are unresolved
          // materials left after this pass
        }
      }
      if (open_geometry) {
        det.init();
        ns.addVolume(det.worldVolume());
      }
      for (xml::Document d : res.includes) {
        print_doc((doc = d).root());
        xml_coll_t(d.root(), DD_CMU(RotationSection)).for_each(Converter<RotationSection>(det, &context));
      }
      for (xml::Document d : res.includes) {
        print_doc((doc = d).root());
        xml_coll_t(d.root(), DD_CMU(SolidSection)).for_each(Converter<SolidSection>(det, &context));
      }
      for (xml::Document d : res.includes) {
        print_doc((doc = d).root());
        xml_coll_t(d.root(), DD_CMU(LogicalPartSection)).for_each(Converter<LogicalPartSection>(det, &context));
      }
      for (xml::Document d : res.includes) {
        print_doc((doc = d).root());
        xml_coll_t(d.root(), DD_CMU(Algorithm)).for_each(Converter<DDLAlgorithm>(det, &context));
      }
      for (xml::Document d : res.includes) {
        print_doc((doc = d).root());
        xml_coll_t(d.root(), DD_CMU(PosPartSection)).for_each(Converter<PosPartSection>(det, &context));
      }
      for (xml::Document d : res.includes) {
        print_doc((doc = d).root());
        xml_coll_t(d.root(), DD_CMU(SpecParSection)).for_each(Converter<SpecParSection>(det, &context));
      }

      /// Unload all XML files after processing
      for (xml::Document d : res.includes)
        Converter<include_unload>(det, &context, &res)(d.root());

      print_doc((doc = dddef.document()).root());
      // Now process the actual geometry items
      xml_coll_t(dddef, DD_CMU(SolidSection)).for_each(Converter<SolidSection>(det, &context));
      {
        // Before we continue, we have to resolve all shapes NOW!
        // Note: This only happens in a legacy DB payloads where
        // boolean shapes can be defined before thier
        // component shapes

        while (!context.unresolvedShapes.empty()) {
          for (auto it = context.unresolvedShapes.begin(); it != context.unresolvedShapes.end();) {
            auto const& name = it->first;
            auto const& aname = std::visit([](auto&& arg) -> std::string { return arg.firstSolidName; }, it->second);
            auto const& bname = std::visit([](auto&& arg) -> std::string { return arg.secondSolidName; }, it->second);

            auto const& ait = context.shapes.find(aname);
            if (ait->second.isValid()) {
              auto const& bit = context.shapes.find(bname);
              if (bit->second.isValid()) {
                dd4hep::Solid shape =
                    std::visit([&ait, &bit](auto&& arg) -> dd4hep::Solid { return arg.make(ait->second, bit->second); },
                               it->second);
                context.shapes[name] = shape;
                it = context.unresolvedShapes.erase(it);
              } else
                ++it;
            } else
              ++it;
          }
        }
      }
      xml_coll_t(dddef, DD_CMU(LogicalPartSection)).for_each(Converter<LogicalPartSection>(det, &context));
      xml_coll_t(dddef, DD_CMU(Algorithm)).for_each(Converter<DDLAlgorithm>(det, &context));
      xml_coll_t(dddef, DD_CMU(PosPartSection)).for_each(Converter<PosPartSection>(det, &context));
      xml_coll_t(dddef, DD_CMU(SpecParSection)).for_each(Converter<SpecParSection>(det, &context));
    } catch (const exception& e) {
      printout(ERROR, "DD4CMS", "Exception while processing xml source:%s", doc.uri().c_str());
      printout(ERROR, "DD4CMS", "----> %s", e.what());
      throw;
    }

    /// This should be the end of all processing....close the geometry
    if (close_geometry) {
      Volume wv = det.worldVolume();
      Volume geomv = ns.volume("cms:OCMS", false);
      if (geomv.isValid())
        wv.placeVolume(geomv, 1);
      Volume mfv = ns.volume("cmsMagneticField:MAGF", false);
      if (mfv.isValid())
        wv.placeVolume(mfv, 1);
      Volume mfv1 = ns.volume("MagneticFieldVolumes:MAGF", false);
      if (mfv1.isValid())
        wv.placeVolume(mfv1, 1);

      det.endDocument();
    }
    printout(INFO, "DDDefinition", "+++ Finished processing %s", fname.c_str());
    return 1;
  }
  except("DDDefinition", "+++ FAILED to process unknown DOM tree [Invalid Handle]");
  return 0;
}

// Now declare the factory entry for the plugin mechanism
DECLARE_XML_DOC_READER(DDDefinition, load_dddefinition)
