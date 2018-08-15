#include "DD4hep/DetFactoryHelper.h"
#include "DD4hep/DetectorHelper.h"
#include "DD4hep/DD4hepUnits.h"
#include "DD4hep/GeoHandler.h"
#include "DD4hep/Printout.h"
#include "DD4hep/Plugins.h"
#include "DD4hep/detail/SegmentationsInterna.h"
#include "DD4hep/detail/DetectorInterna.h"
#include "DD4hep/detail/ObjectsInterna.h"

#include "XML/Utilities.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DetectorDescription/DDCMS/interface/DDAlgoArguments.h"
#include "DetectorDescription/DDCMS/interface/DDNamespace.h"
#include "DetectorDescription/DDCMS/interface/DDParsingContext.h"
#include "DetectorDescription/DDCMS/interface/DDRegistry.h"

#include "TGeoManager.h"
#include "TGeoMaterial.h"

#include <climits>
#include <iostream>
#include <iomanip>
#include <set>
#include <map>
#include <utility>

using namespace std;
using namespace dd4hep;
using namespace cms;

namespace dd4hep {

  namespace {
    
    UInt_t unique_mat_id = 0xAFFEFEED;

    class disabled_algo;
    class include_constants;
    class include_load;
    class include_unload;
    class print_xml_doc;
    
    class ConstantsSection;
    class DDLConstant;
    class DDRegistry {
    public:
      std::vector<xml::Document> includes;
      std::map<std::string, std::string> unresolvedConst, allConst, originalConst;
    };

    class MaterialSection;
    class DDLElementaryMaterial;
    class DDLCompositeMaterial;
  
    class RotationSection;
    class DDLRotation;
    class DDLRotationSequence;
    class DDLRotationByAxis;
    class DDLTransform3D;

    class PosPartSection;
    class DDLPosPart;

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
    
    class vissection;
    class vis;
    class debug;
  }

  /// Converter instances implemented in this compilation unit
  template <> void Converter<debug>::operator()(xml_h element) const;
  template <> void Converter<print_xml_doc>::operator()(xml_h element) const;
  template <> void Converter<disabled_algo>::operator()(xml_h element) const;
  
  /// Converter for <ConstantsSection/> tags
  template <> void Converter<ConstantsSection>::operator()(xml_h element) const;
  template <> void Converter<DDLConstant>::operator()(xml_h element) const;
  template <> void Converter<DDRegistry>::operator()(xml_h element) const;

  /// Converter for <VisSection/> tags
  template <> void Converter<vissection>::operator()(xml_h element) const;
  /// Convert compact visualization attributes
  template <> void Converter<vis>::operator()(xml_h element) const;

  /// Converter for <MaterialSection/> tags
  template <> void Converter<MaterialSection>::operator()(xml_h element) const;
  template <> void Converter<DDLElementaryMaterial>::operator()(xml_h element) const;
  template <> void Converter<DDLCompositeMaterial>::operator()(xml_h element) const;

  /// Converter for <RotationSection/> tags
  template <> void Converter<RotationSection>::operator()(xml_h element) const;
  /// Converter for <DDLRotation/> tags
  template <> void Converter<DDLRotation>::operator()(xml_h element) const;
  /// Converter for <DDLRotationSequence/> tags
  template <> void Converter<DDLRotationSequence>::operator()(xml_h element) const;
  /// Converter for <DDLRotationByAxis/> tags
  template <> void Converter<DDLRotationByAxis>::operator()(xml_h element) const;
  template <> void Converter<DDLTransform3D>::operator()(xml_h element) const;

  /// Generic converter for  <LogicalPartSection/> tags
  template <> void Converter<LogicalPartSection>::operator()(xml_h element) const;
  template <> void Converter<DDLLogicalPart>::operator()(xml_h element) const;

  template <> void Converter<PosPartSection>::operator()(xml_h element) const;
  /// Converter for <PosPart/> tags
  template <> void Converter<DDLPosPart>::operator()(xml_h element) const;
  
  /// Generic converter for solids: <SolidSection/> tags
  template <> void Converter<SolidSection>::operator()(xml_h element) const;
  /// Converter for <UnionSolid/> tags
  template <> void Converter<DDLUnionSolid>::operator()(xml_h element) const;
  /// Converter for <SubtractionSolid/> tags
  template <> void Converter<DDLSubtractionSolid>::operator()(xml_h element) const;
  /// Converter for <IntersectionSolid/> tags
  template <> void Converter<DDLIntersectionSolid>::operator()(xml_h element) const;
  /// Converter for <PseudoTrap/> tags
  template <> void Converter<DDLPseudoTrap>::operator()(xml_h element) const;
  /// Converter for <ExtrudedPolygon/> tags
  template <> void Converter<DDLExtrudedPolygon>::operator()(xml_h element) const;
  /// Converter for <ShapelessSolid/> tags
  template <> void Converter<DDLShapeless>::operator()(xml_h element) const;
  /// Converter for <Trapezoid/> tags
  template <> void Converter<DDLTrapezoid>::operator()(xml_h element) const;
  /// Converter for <Polycone/> tags
  template <> void Converter<DDLPolycone>::operator()(xml_h element) const;
  /// Converter for <Polyhedra/> tags
  template <> void Converter<DDLPolyhedra>::operator()(xml_h element) const;
  /// Converter for <EllipticalTube/> tags
  template <> void Converter<DDLEllipticalTube>::operator()(xml_h element) const;
  /// Converter for <Torus/> tags
  template <> void Converter<DDLTorus>::operator()(xml_h element) const;
  /// Converter for <Tubs/> tags
  template <> void Converter<DDLTubs>::operator()(xml_h element) const;
  /// Converter for <CutTubs/> tags
  template <> void Converter<DDLCutTubs>::operator()(xml_h element) const;
  /// Converter for <TruncTubs/> tags
  template <> void Converter<DDLTruncTubs>::operator()(xml_h element) const;
  /// Converter for <Sphere/> tags
  template <> void Converter<DDLSphere>::operator()(xml_h element) const;
  /// Converter for <Trd1/> tags
  template <> void Converter<DDLTrd1>::operator()(xml_h element) const;
  /// Converter for <Cone/> tags
  template <> void Converter<DDLCone>::operator()(xml_h element) const;
  /// Converter for <DDLBox/> tags
  template <> void Converter<DDLBox>::operator()(xml_h element) const;
  /// Converter for <Algorithm/> tags
  template <> void Converter<DDLAlgorithm>::operator()(xml_h element) const;
  /// Converter for <Vector/> tags
  template <> void Converter<DDLVector>::operator()(xml_h element) const;

  /// DD4hep specific: Load include file
  template <> void Converter<include_load>::operator()(xml_h element) const;
  /// DD4hep specific: Unload include file
  template <> void Converter<include_unload>::operator()(xml_h element) const;
  /// DD4hep specific: Process constants objects
  template <> void Converter<include_constants>::operator()(xml_h element) const;
}

/// Converter for <ConstantsSection/> tags
template <> void Converter<ConstantsSection>::operator()(xml_h element) const  {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>(), element);
  xml_coll_t(element, _CMU(Constant)).for_each(Converter<DDLConstant>(description,_ns.context,optional));
  xml_coll_t(element, _CMU(Vector)).for_each(Converter<DDLVector>(description,_ns.context,optional));
}

/// Converter for <VisSection/> tags
template <> void Converter<vissection>::operator()(xml_h element) const  {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>(), element);
  xml_coll_t(element, _CMU(vis)).for_each(Converter<vis>(description,_ns.context,optional));
}

/// Converter for <MaterialSection/> tags
template <> void Converter<MaterialSection>::operator()(xml_h element) const   {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>(), element);
  xml_coll_t(element, _CMU(ElementaryMaterial)).for_each(Converter<DDLElementaryMaterial>(description,_ns.context,optional));
  xml_coll_t(element, _CMU(CompositeMaterial)).for_each(Converter<DDLCompositeMaterial>(description,_ns.context,optional));
}

template <> void Converter<RotationSection>::operator()(xml_h element) const   {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>(), element);
  xml_coll_t(element, _CMU(Rotation)).for_each(Converter<DDLRotation>(description,_ns.context,optional));
  xml_coll_t(element, _CMU(RotationSequence)).for_each(Converter<DDLRotationSequence>(description,_ns.context,optional));
  xml_coll_t(element, _CMU(RotationByAxis)).for_each(Converter<DDLRotationByAxis>(description,_ns.context,optional));
}

template <> void Converter<PosPartSection>::operator()(xml_h element) const   {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>(), element);
  xml_coll_t(element, _CMU(PosPart)).for_each(Converter<DDLPosPart>(description,_ns.context,optional));
}

/// Generic converter for  <LogicalPartSection/> tags
template <> void Converter<LogicalPartSection>::operator()(xml_h element) const   {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>(), element);
  xml_coll_t(element, _CMU(LogicalPart)).for_each(Converter<DDLLogicalPart>(description,_ns.context,optional));
}

template <> void Converter<disabled_algo>::operator()(xml_h element) const   {
  cms::DDParsingContext* c = _param<cms::DDParsingContext>();
  c->disabledAlgs.insert(element.attr<string>(_U(name)));
}

/// Generic converter for  <SolidSection/> tags
template <> void Converter<SolidSection>::operator()(xml_h element) const   {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>(), element);
  for(xml_coll_t solid(element, _U(star)); solid; ++solid)   {
    string tag = solid.tag();
    using cms::hash;
    switch( hash( solid.tag()))
    {
    case hash("Box"):
      Converter<DDLBox>(description,_ns.context,optional)(solid);
      break;
    case hash("Polycone"):
      Converter<DDLPolycone>(description,_ns.context,optional)(solid);
      break;
    case hash("Polyhedra"):
      Converter<DDLPolyhedra>(description,_ns.context,optional)(solid);
      break;
    case hash("Tubs"):
      Converter<DDLTubs>(description,_ns.context,optional)(solid);
      break;
    case hash("CutTubs"):
      Converter<DDLCutTubs>(description,_ns.context,optional)(solid);
      break;
    case hash("TruncTubs"):
      Converter<DDLTruncTubs>(description,_ns.context,optional)(solid);
      break;
    case hash("Tube"):
      Converter<DDLTubs>(description,_ns.context,optional)(solid);
      break;
    case hash("Trd1"):
      Converter<DDLTrd1>(description,_ns.context,optional)(solid);
      break;
    case hash("Cone"):
      Converter<DDLCone>(description,_ns.context,optional)(solid);
      break;
    case hash("Sphere"):
      Converter<DDLSphere>(description,_ns.context,optional)(solid);
      break;
    case hash("EllipticalTube"):
      Converter<DDLEllipticalTube>(description,_ns.context,optional)(solid);
      break;
    case hash("Torus"):
      Converter<DDLTorus>(description,_ns.context,optional)(solid);
      break;
    case hash("PseudoTrap"):
      Converter<DDLPseudoTrap>(description,_ns.context,optional)(solid);
      break;
    case hash("ExtrudedPolygon"):
      Converter<DDLExtrudedPolygon>(description,_ns.context,optional)(solid);
      break;
    case hash("Trapezoid"):
      Converter<DDLTrapezoid>(description,_ns.context,optional)(solid);
      break;
    case hash("UnionSolid"):
      Converter<DDLUnionSolid>(description,_ns.context,optional)(solid);
      break;
    case hash("SubtractionSolid"):
      Converter<DDLSubtractionSolid>(description,_ns.context,optional)(solid);
      break;
    case hash("IntersectionSolid"):
      Converter<DDLIntersectionSolid>(description,_ns.context,optional)(solid);
      break;
    case hash("ShapelessSolid"):
      Converter<DDLShapeless>(description,_ns.context,optional)(solid);
      break;
    default:
      throw std::runtime_error( "Request to process unknown shape '" + xml_dim_t(solid).nameStr() + "' [" + tag + "]");
      break;
    }
  }
}

/// Converter for <Constant/> tags
template <> void Converter<DDLConstant>::operator()(xml_h element) const  {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>());
  DDRegistry*  res  = _option<DDRegistry>();
  xml_dim_t constant = element;
  xml_dim_t par  = constant.parent();
  bool      eval = par.hasAttr(_U(eval)) ? par.attr<bool>(_U(eval)) : false;
  string    val  = constant.valueStr();
  string    nam  = constant.nameStr();
  string    real = _ns.prepend(nam);
  string    typ  = eval ? "number" : "string";
  size_t    idx  = val.find('[');
  
  if ( constant.hasAttr(_U(type)) )
    typ = constant.typeStr();

  if ( idx == string::npos || typ == "string" )  {
    try  {
      _ns.addConstant(nam, val, typ);
      res->allConst[real] = val;
      res->originalConst[real] = val;
    }
    catch(const exception& e)   {
      printout(INFO,"MyDDCMS","++ Unresolved constant: %s = %s [%s]. Try to resolve later. [%s]",
               real.c_str(), val.c_str(), typ.c_str(), e.what());
    }
    return;
  }
  // Setup the resolution mechanism in Converter<resolve>
  while ( idx != string::npos )  {
    ++idx;
    size_t idp = val.find(':',idx);
    size_t idq = val.find(']',idx);
    if ( idp == string::npos || idp > idq )
      val.insert(idx,_ns.name());
    else if ( idp != string::npos && idp < idq )
      val[idp] = NAMESPACE_SEP;
    idx = val.find('[',idx);
  }
  printout(_ns.context->debug_constants ? ALWAYS : DEBUG,
           "Constant","Unresolved: %s -> %s",real.c_str(),val.c_str());
  res->allConst[real] = val;
  res->originalConst[real] = val;
  res->unresolvedConst[real] = val;
}

/** Convert compact visualization attribute to Detector visualization attribute
 *
 *  <vis name="SiVertexBarrelModuleVis"
 *       alpha="1.0" r="1.0" g="0.75" b="0.76"
 *       drawingStyle="wireframe"
 *       showDaughters="false"
 *       visible="true"/>
 */
template <> void Converter<vis>::operator()(xml_h e) const {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>());
  VisAttr attr(e.attr<string>(_U(name)));
  float red   = e.hasAttr(_U(r)) ? e.attr<float>(_U(r)) : 1.0f;
  float green = e.hasAttr(_U(g)) ? e.attr<float>(_U(g)) : 1.0f;
  float blue  = e.hasAttr(_U(b)) ? e.attr<float>(_U(b)) : 1.0f;

  printout(_ns.context->debug_visattr ? ALWAYS : DEBUG, "Compact",
           "++ Converting VisAttr  structure: %-16s. R=%.3f G=%.3f B=%.3f",
           attr.name(), red, green, blue);
  attr.setColor(red, green, blue);
  if (e.hasAttr(_U(alpha)))
    attr.setAlpha(e.attr<float>(_U(alpha)));
  if (e.hasAttr(_U(visible)))
    attr.setVisible(e.attr<bool>(_U(visible)));
  if (e.hasAttr(_U(lineStyle))) {
    string ls = e.attr<string>(_U(lineStyle));
    if (ls == "unbroken")
      attr.setLineStyle(VisAttr::SOLID);
    else if (ls == "broken")
      attr.setLineStyle(VisAttr::DASHED);
  }
  else {
    attr.setLineStyle(VisAttr::SOLID);
  }
  if (e.hasAttr(_U(drawingStyle))) {
    string ds = e.attr<string>(_U(drawingStyle));
    if (ds == "wireframe")
      attr.setDrawingStyle(VisAttr::WIREFRAME);
    else if (ds == "solid")
      attr.setDrawingStyle(VisAttr::SOLID);
  }
  else {
    attr.setDrawingStyle(VisAttr::SOLID);
  }
  if (e.hasAttr(_U(showDaughters)))
    attr.setShowDaughters(e.attr<bool>(_U(showDaughters)));
  else
    attr.setShowDaughters(true);
  description.addVisAttribute(attr);
}

/// Converter for <DDLElementaryMaterial/> tags
template <> void Converter<DDLElementaryMaterial>::operator()(xml_h element) const   {
  cms::DDNamespace     _ns(_param<cms::DDParsingContext>());
  xml_dim_t     xmat(element);
  string        nam = _ns.prepend(xmat.nameStr());
  TGeoManager&  mgr = description.manager();
  TGeoMaterial* mat = mgr.GetMaterial(nam.c_str());
  if ( nullptr == mat )   {
    const char* matname = nam.c_str();
    double density      = xmat.density();
    //double atomicWeight = xmat.attr<double>(_CMU(atomicWeight));
    double atomicNumber = xmat.attr<double>(_CMU(atomicNumber));
    TGeoElementTable* tab = mgr.GetElementTable();
    TGeoMixture*      mix = new TGeoMixture(nam.c_str(), 1, density);
    TGeoElement*      elt = tab->FindElement(xmat.nameStr().c_str());

    printout(_ns.context->debug_materials ? ALWAYS : DEBUG, "MyDDCMS",
             "+++ Converting material %-48s  Density: %.3f.",
             ('"'+nam+'"').c_str(), density);

    if ( !elt )  {
      printout(WARNING,"MyDDCMS",
               "+++ Converter<ElementaryMaterial> No element present with name:%s  [FAKE IT]",
               matname);
      int n = int(atomicNumber/2e0);
      if ( n < 2 ) n = 2;
      elt = new TGeoElement(xmat.nameStr().c_str(),"CMS element",n,atomicNumber);
      //return;
    }
    if ( elt->Z() == 0 )   {
      int n = int(atomicNumber/2e0);
      if ( n < 2 ) n = 2;
      elt = new TGeoElement((xmat.nameStr()+"-CMS").c_str(),"CMS element",n,atomicNumber);
    }
    mix->AddElement(elt, 1.0);
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

/// Converter for <DDLCompositeMaterial/> tags
template <> void Converter<DDLCompositeMaterial>::operator()(xml_h element) const   {
  cms::DDNamespace     _ns(_param<cms::DDParsingContext>());
  xml_dim_t     xmat(element);
  string        nam = _ns.prepend(xmat.nameStr());
  TGeoManager&  mgr = description.manager();
  TGeoMaterial* mat = mgr.GetMaterial(nam.c_str());
  if ( nullptr == mat )   {
    const char*  matname = nam.c_str();
    double       density = xmat.density();
    xml_coll_t   composites(xmat,_CMU(MaterialFraction));
    TGeoMixture* mix = new TGeoMixture(nam.c_str(), composites.size(), density);

    printout(_ns.context->debug_materials ? ALWAYS : DEBUG, "MyDDCMS",
             "++ Converting material %-48s  Density: %.3f.",
             ('"'+nam+'"').c_str(), density);
    
    for (composites.reset(); composites; ++composites)   {
      xml_dim_t xfrac(composites);
      xml_dim_t xfrac_mat(xfrac.child(_CMU(rMaterial)));
      double    fraction = xfrac.fraction();
      string    fracname = _ns.realName(xfrac_mat.nameStr());

      TGeoMaterial* frac_mat = mgr.GetMaterial(fracname.c_str());
      if ( frac_mat )  {
        mix->AddElement(frac_mat, fraction);
        continue;
      }
      printout(WARNING,"MyDDCMS","+++ Composite material \"%s\" not present!",
               fracname.c_str());
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
template <> void Converter<DDLRotation>::operator()(xml_h element) const  {
  cms::DDParsingContext* context = _param<cms::DDParsingContext>();
  cms::DDNamespace _ns(context);
  xml_dim_t xrot(element);
  string    nam    = xrot.nameStr();
  double    thetaX = xrot.hasAttr(_CMU(thetaX)) ? _ns.attr<double>(xrot,_CMU(thetaX)) : 0e0;
  double    phiX   = xrot.hasAttr(_CMU(phiX))   ? _ns.attr<double>(xrot,_CMU(phiX))   : 0e0;
  double    thetaY = xrot.hasAttr(_CMU(thetaY)) ? _ns.attr<double>(xrot,_CMU(thetaY)) : 0e0;
  double    phiY   = xrot.hasAttr(_CMU(phiY))   ? _ns.attr<double>(xrot,_CMU(phiY))   : 0e0;
  double    thetaZ = xrot.hasAttr(_CMU(thetaZ)) ? _ns.attr<double>(xrot,_CMU(thetaZ)) : 0e0;
  double    phiZ   = xrot.hasAttr(_CMU(phiZ))   ? _ns.attr<double>(xrot,_CMU(phiZ))   : 0e0;
  Rotation3D rot = makeRotation3D(thetaX, phiX, thetaY, phiY, thetaZ, phiZ);
  printout(context->debug_rotations ? ALWAYS : DEBUG,
           "MyDDCMS","+++ Adding rotation: %-32s: (theta/phi)[rad] X: %6.3f %6.3f Y: %6.3f %6.3f Z: %6.3f %6.3f",
           _ns.prepend(nam).c_str(),thetaX,phiX,thetaY,phiY,thetaZ,phiZ);
  _ns.addRotation(nam, rot);
}

/// Converter for <RotationSequence/> tags
template <> void Converter<DDLRotationSequence>::operator()(xml_h element) const {
  cms::DDParsingContext* context = _param<cms::DDParsingContext>();
  cms::DDNamespace _ns(context);
  xml_dim_t xrot(element);
  string nam = xrot.nameStr();
  Rotation3D rot;
  xml_coll_t rotations(xrot,_CMU(RotationByAxis));
  for( rotations.reset(); rotations; ++rotations ) {
    string    axis = _ns.attr<string>(rotations,_CMU(axis));
    double    angle = _ns.attr<double>(rotations,_U(angle));
    rot = makeRotation3D( rot, axis, angle );
    printout(context->debug_rotations ? ALWAYS : DEBUG,
	     "MyDDCMS","+   Adding rotation to: %-29s: (axis/angle)[rad] Axis: %s Angle: %6.3f",
	     nam.c_str(), axis.c_str(), angle);
  }
  double xx, xy, xz;
  double yx, yy, yz;
  double zx, zy, zz;
  rot.GetComponents(xx,xy,xz,yx,yy,yz,zx,zy,zz);
  printout(context->debug_rotations ? ALWAYS : DEBUG,
	   "MyDDCMS","+++ Adding rotation sequence: %-23s: %6.3f %6.3f %6.3f, %6.3f %6.3f %6.3f, %6.3f %6.3f %6.3f",
	   _ns.prepend(nam).c_str(), xx, xy, xz, yx, yy, yz, zx, zy, zz);
  _ns.addRotation(nam, rot);
}

/// Converter for <RotationByAxis/> tags
template <> void Converter<DDLRotationByAxis>::operator()(xml_h element) const  {
  cms::DDParsingContext* context = _param<cms::DDParsingContext>();
  cms::DDNamespace _ns(context);
  xml_dim_t xrot(element);
  xml_dim_t par(xrot.parent());
  if( xrot.hasAttr(_U(name))) {
    string    nam  = xrot.nameStr();// + string("Rotation"); // xrot.hasAttr(_U(name)) ? xrot.nameStr() : par.nameStr();
    string    axis = _ns.attr<string>(xrot,_CMU(axis));
    double    angle = _ns.attr<double>(xrot,_U(angle));
    Rotation3D rot;
    rot = makeRotation3D( rot, axis, angle );
    printout(context->debug_rotations ? ALWAYS : DEBUG,
	     "MyDDCMS","+++ Adding rotation: %-32s: (axis/angle)[rad] Axis: %s Angle: %6.3f",
	     _ns.prepend(nam).c_str(), axis.c_str(), angle);
    _ns.addRotation(nam, rot);
  }
}

/// Converter for <LogicalPart/> tags
template <> void Converter<DDLLogicalPart>::operator()(xml_h element) const {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string    sol = e.child(_CMU(rSolid)).attr<string>(_U(name));
  string    mat = e.child(_CMU(rMaterial)).attr<string>(_U(name));
  _ns.addVolume(Volume(e.nameStr(), _ns.solid(sol), _ns.material(mat)));
}

/// Helper converter
template <> void Converter<DDLTransform3D>::operator()(xml_h element) const {
  cms::DDNamespace    _ns(_param<cms::DDParsingContext>());
  Transform3D* tr = _option<Transform3D>();
  xml_dim_t   e(element);
  xml_dim_t   translation = e.child(_CMU(Translation),false);
  xml_dim_t   rotation    = e.child(_CMU(Rotation),false);
  xml_dim_t   refRotation = e.child(_CMU(rRotation),false);
  Position    pos;
  Rotation3D  rot;

  if ( translation.ptr() )   {
    double x = _ns.attr<double>(translation,_U(x));
    double y = _ns.attr<double>(translation,_U(y));
    double z = _ns.attr<double>(translation,_U(z));
    pos = Position(x,y,z);
  }
  if ( rotation.ptr() )   {
    double x = _ns.attr<double>(rotation,_U(x));
    double y = _ns.attr<double>(rotation,_U(y));
    double z = _ns.attr<double>(rotation,_U(z));
    rot = RotationZYX(z,y,x);
  }
  else if ( refRotation.ptr() )   {
    rot = _ns.rotation(refRotation.nameStr());
  }
  *tr = Transform3D(rot,pos);
}

/// Converter for <PosPart/> tags
template <> void Converter<DDLPosPart>::operator()(xml_h element) const {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>());
  xml_dim_t   e(element);
  int         copy        = e.attr<int>(_CMU(copyNumber));
  string      parent_nam  = _ns.attr<string>(e.child(_CMU(rParent)),_U(name));
  string      child_nam   = _ns.attr<string>(e.child(_CMU(rChild)),_U(name));
  Volume      parent      = _ns.volume(parent_nam);
  Volume      child       = _ns.volume(child_nam, false);
  
  printout(_ns.context->debug_placements ? ALWAYS : DEBUG, "MyDDCMS",
           "+++ %s Parent: %-24s [%s] Child: %-32s [%s] copy:%d",
           e.tag().c_str(),
           parent_nam.c_str(), parent.isValid() ? "VALID" : "INVALID",
           child_nam.c_str(),  child.isValid()  ? "VALID" : "INVALID",
           copy);
  PlacedVolume pv;
  if ( child.isValid() )   {
    Transform3D trafo;
    Converter<DDLTransform3D>(description,param,&trafo)(element);
    pv = parent.placeVolume(child,copy,trafo);
  }
  if ( !pv.isValid() )   {
    printout(ERROR,"MyDDCMS","+++ Placement FAILED! Parent:%s Child:%s Valid:%s",
             parent.name(), child_nam.c_str(), yes_no(child.isValid()));
  }
}

template <typename TYPE>
static void convert_boolean(cms::DDParsingContext* context, xml_h element)   {
  cms::DDNamespace   _ns(context);
  xml_dim_t   e(element);
  string      nam = e.nameStr();
  Solid       solids[2];
  Solid       boolean;
  int cnt = 0;

  if ( e.hasChild(_CMU(rSolid)) )  {   // Old version
    for(xml_coll_t c(element, _CMU(rSolid)); cnt<2 && c; ++c, ++cnt)
      solids[cnt] = _ns.solid(c.attr<string>(_U(name)));
  }
  else  {
    if ( (solids[0] = _ns.solid(e.attr<string>(_CMU(firstSolid)))).isValid() ) ++cnt;
    if ( (solids[1] = _ns.solid(e.attr<string>(_CMU(secondSolid)))).isValid() ) ++cnt;
  }
  if ( cnt != 2 )   {
    except("MyDDCMS","+++ Failed to create boolean solid %s. Found only %d parts.",nam.c_str(), cnt);
  }
  printout(_ns.context->debug_placements ? ALWAYS : DEBUG, "MyDDCMS",
           "+++ BooleanSolid: %s Left: %-32s Right: %-32s",
           nam.c_str(), solids[0]->GetName(), solids[1]->GetName());

  if ( solids[0].isValid() && solids[1].isValid() )  {
    Transform3D trafo;
    Converter<DDLTransform3D>(*context->description,context,&trafo)(element);
    boolean = TYPE(solids[0],solids[1],trafo);
  }
  if ( !boolean.isValid() )
    except("MyDDCMS","+++ FAILED to construct subtraction solid: %s",nam.c_str());
  _ns.addSolid(nam,boolean);
}

/// Converter for <SubtractionSolid/> tags
template <> void Converter<DDLUnionSolid>::operator()(xml_h element) const   {
  convert_boolean<UnionSolid>(_param<cms::DDParsingContext>(),element);
}

/// Converter for <SubtractionSolid/> tags
template <> void Converter<DDLSubtractionSolid>::operator()(xml_h element) const   {
  convert_boolean<SubtractionSolid>(_param<cms::DDParsingContext>(),element);
}

/// Converter for <SubtractionSolid/> tags
template <> void Converter<DDLIntersectionSolid>::operator()(xml_h element) const   {
  convert_boolean<IntersectionSolid>(_param<cms::DDParsingContext>(),element);
}

/// Converter for <Polycone/> tags
template <> void Converter<DDLPolycone>::operator()(xml_h element) const {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam      = e.nameStr();
  double startPhi = _ns.attr<double>(e,_CMU(startPhi));
  double deltaPhi = _ns.attr<double>(e,_CMU(deltaPhi));
  vector<double> z, rmin, rmax, r;
  
  for(xml_coll_t rzpoint(element, _CMU(RZPoint)); rzpoint; ++rzpoint) {
    z.emplace_back(_ns.attr<double>(rzpoint,_U(z)));
    r.emplace_back(_ns.attr<double>(rzpoint,_U(r)));
  }
  if( z.empty()) {
    for(xml_coll_t zplane(element, _CMU(ZSection)); zplane; ++zplane)   {
      rmin.emplace_back(_ns.attr<double>(zplane,_CMU(rMin)));
      rmax.emplace_back(_ns.attr<double>(zplane,_CMU(rMax)));
      z.emplace_back(_ns.attr<double>(zplane,_U(z)));
    }
    printout(_ns.context->debug_shapes ? ALWAYS : DEBUG, "MyDDCMS",
	     "+   Polycone: startPhi=%10.3f [rad] deltaPhi=%10.3f [rad]  %4ld z-planes",
	     startPhi, deltaPhi, z.size());
    _ns.addSolid(nam, Polycone(startPhi,deltaPhi,rmin,rmax,z));
  }
  else {
    printout(_ns.context->debug_shapes ? ALWAYS : DEBUG, "MyDDCMS",
	     "+   Polycone: startPhi=%10.3f [rad] deltaPhi=%10.3f [rad]  %4ld z-planes and %4ld radii",
	     startPhi, deltaPhi, z.size(), r.size());
    _ns.addSolid(nam, Polycone(startPhi,deltaPhi,r,z));
  }
}

/// Converter for <ExtrudedPolygon/> tags
template <> void Converter<DDLExtrudedPolygon>::operator()(xml_h element) const  {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam      = e.nameStr();
  vector<double> pt_x, pt_y, sec_x, sec_y, sec_z, sec_scale;
  
  for(xml_coll_t sec(element, _CMU(ZXYSection)); sec; ++sec)   {
    sec_z.emplace_back(_ns.attr<double>(sec,_U(z)));
    sec_x.emplace_back(_ns.attr<double>(sec,_U(x)));
    sec_y.emplace_back(_ns.attr<double>(sec,_U(y)));
    sec_scale.emplace_back(_ns.attr<double>(sec,_CMU(scale),1.0));
  }
  for(xml_coll_t pt(element, _CMU(XYPoint)); pt; ++pt)   {
    pt_x.emplace_back(_ns.attr<double>(pt,_U(x)));
    pt_y.emplace_back(_ns.attr<double>(pt,_U(y)));
  }
  printout(_ns.context->debug_shapes ? ALWAYS : DEBUG, "MyDDCMS",
	   "+   ExtrudedPolygon: %4ld points %4ld zxy sections",
	   pt_x.size(), sec_z.size());
  _ns.addSolid(nam,ExtrudedPolygon(pt_x,pt_y,sec_z,sec_x,sec_y,sec_scale));
}

/// Converter for <Polyhedra/> tags
template <> void Converter<DDLPolyhedra>::operator()(xml_h element) const {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam      = e.nameStr();
  double numSide  = _ns.attr<int>(e,_CMU(numSide));
  double startPhi = _ns.attr<double>(e,_CMU(startPhi));
  double deltaPhi = _ns.attr<double>(e,_CMU(deltaPhi));
  vector<double> z, rmin, rmax;
  
  for(xml_coll_t zplane(element, _CMU(RZPoint)); zplane; ++zplane)   {
    rmin.emplace_back(0.0);
    rmax.emplace_back(_ns.attr<double>(zplane,_U(r)));
    z.emplace_back(_ns.attr<double>(zplane,_U(z)));
  }
  for(xml_coll_t zplane(element, _CMU(ZSection)); zplane; ++zplane)   {
    rmin.emplace_back(_ns.attr<double>(zplane,_CMU(rMin)));
    rmax.emplace_back(_ns.attr<double>(zplane,_CMU(rMax)));
    z.emplace_back(_ns.attr<double>(zplane,_U(z)));
  }
  printout(_ns.context->debug_shapes ? ALWAYS : DEBUG, "MyDDCMS",
	   "+   Polyhedra:startPhi=%8.3f [rad] deltaPhi=%8.3f [rad]  %4d sides %4ld z-planes",
	   startPhi, deltaPhi, numSide, z.size());
  _ns.addSolid(nam, Polyhedra(numSide,startPhi,deltaPhi,z,rmin,rmax));
}

/// Converter for <Sphere/> tags
template <> void Converter<DDLSphere>::operator()(xml_h element) const {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam      = e.nameStr();
  double rinner   = _ns.attr<double>(e,_CMU(innerRadius));
  double router   = _ns.attr<double>(e,_CMU(outerRadius));
  double startPhi = _ns.attr<double>(e,_CMU(startPhi));
  double deltaPhi = _ns.attr<double>(e,_CMU(deltaPhi));
  double startTheta = _ns.attr<double>(e,_CMU(startTheta));
  double deltaTheta = _ns.attr<double>(e,_CMU(deltaTheta));
  printout(_ns.context->debug_shapes ? ALWAYS : DEBUG, "MyDDCMS",
	   "+   Sphere:   r_inner=%8.3f [cm] r_outer=%8.3f [cm]"
	   " startPhi=%8.3f [rad] deltaPhi=%8.3f startTheta=%8.3f delteTheta=%8.3f [rad]",
	   rinner, router, startPhi, deltaPhi, startTheta, deltaTheta);
  _ns.addSolid(nam, Sphere(rinner, router, startTheta, deltaTheta, startPhi, deltaPhi));
}

/// Converter for <Torus/> tags
template <> void Converter<DDLTorus>::operator()(xml_h element) const   {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam      = e.nameStr();
  double r        = _ns.attr<double>(e,_CMU(torusRadius));
  double rinner   = _ns.attr<double>(e,_CMU(innerRadius));
  double router   = _ns.attr<double>(e,_CMU(outerRadius));
  double startPhi = _ns.attr<double>(e,_CMU(startPhi));
  double deltaPhi = _ns.attr<double>(e,_CMU(deltaPhi));
  printout(_ns.context->debug_shapes ? ALWAYS : DEBUG, "MyDDCMS",
           "+   Torus:    r=%10.3f [cm] r_inner=%10.3f [cm] r_outer=%10.3f [cm]"
           " startPhi=%10.3f [rad] deltaPhi=%10.3f [rad]",
           r, rinner, router, startPhi, deltaPhi);
  _ns.addSolid(nam, Torus(r, rinner, router, startPhi, deltaPhi));
}

/// Converter for <Pseudotrap/> tags
template <> void Converter<DDLPseudoTrap>::operator()(xml_h element) const {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam      = e.nameStr();
  double dx1      = _ns.attr<double>(e,_CMU(dx1));
  double dy1      = _ns.attr<double>(e,_CMU(dy1));
  double dx2      = _ns.attr<double>(e,_CMU(dx2));
  double dy2      = _ns.attr<double>(e,_CMU(dy2));
  double dz       = _ns.attr<double>(e,_U(dz));
  double r        = _ns.attr<double>(e,_U(radius));
  bool   atMinusZ = _ns.attr<bool>  (e,_CMU(atMinusZ));
  printout(_ns.context->debug_shapes ? ALWAYS : DEBUG, "MyDDCMS",
	   "+   Pseudotrap:  dz=%8.3f [cm] dx1:%.3f dy1:%.3f dx2=%.3f dy2=%.3f radius:%.3f atMinusZ:%s",
	   dz, dx1, dy1, dx2, dy2, r, yes_no(atMinusZ));
  _ns.addSolid(nam, PseudoTrap(dx1, dx2, dy1, dy2, dz, r, atMinusZ));
}

/// Converter for <Trapezoid/> tags
template <> void Converter<DDLTrapezoid>::operator()(xml_h element) const {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam      = e.nameStr();
  double dz       = _ns.attr<double>(e,_U(dz));
  double alp1     = _ns.attr<double>(e,_CMU(alp1));
  double bl1      = _ns.attr<double>(e,_CMU(bl1));
  double tl1      = _ns.attr<double>(e,_CMU(tl1));
  double h1       = _ns.attr<double>(e,_CMU(h1));
  double alp2     = _ns.attr<double>(e,_CMU(alp2));
  double bl2      = _ns.attr<double>(e,_CMU(bl2));
  double tl2      = _ns.attr<double>(e,_CMU(tl2));
  double h2       = _ns.attr<double>(e,_CMU(h2));
  double phi      = _ns.attr<double>(e,_U(phi),0.0);
  double theta    = _ns.attr<double>(e,_U(theta),0.0);

  printout(_ns.context->debug_shapes ? ALWAYS : DEBUG, "MyDDCMS",
           "+   Trapezoid:  dz=%10.3f [cm] alp1:%.3f bl1=%.3f tl1=%.3f alp2=%.3f bl2=%.3f tl2=%.3f h2=%.3f phi=%.3f theta=%.3f",
           dz, alp1, bl1, tl1, h1, alp2, bl2, tl2, h2, phi, theta);
  _ns.addSolid( nam, Trap( dz, theta, phi, h1, bl1, tl1, alp1, h2, bl2, tl2, alp2 ));
}

/// Converter for <Trd1/> tags
template <> void Converter<DDLTrd1>::operator()(xml_h element) const {
  cms::DDNamespace _ns( _param<cms::DDParsingContext>());
  xml_dim_t e( element );
  string nam      = e.nameStr();
  double dx1      = _ns.attr<double>( e, _CMU( dx1 ));
  double dy1      = _ns.attr<double>( e, _CMU( dy1 ));
  double dx2      = _ns.attr<double>( e, _CMU( dx2 ), 0.0 );
  double dy2      = _ns.attr<double>( e, _CMU( dy2 ), dy1 );
  double dz       = _ns.attr<double>( e, _CMU( dz ));
  printout(_ns.context->debug_shapes ? ALWAYS : DEBUG, "MyDDCMS",
	   "+   Trd1:       dz=%8.3f [cm] dx1:%.3f dy1:%.3f dx2:%.3f dy2:%.3f",
	   dz, dx1, dy1, dx2, dy2);
  _ns.addSolid( nam, Trapezoid( dx1, dx2, dy1, dy2, dz ));
}

/// Converter for <Tubs/> tags
template <> void Converter<DDLTubs>::operator()(xml_h element) const {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam      = e.nameStr();
  double dz       = _ns.attr<double>(e,_CMU(dz));
  double rmin     = _ns.attr<double>(e,_CMU(rMin));
  double rmax     = _ns.attr<double>(e,_CMU(rMax));
  double startPhi = _ns.attr<double>(e,_CMU(startPhi),0.0);
  double deltaPhi = _ns.attr<double>(e,_CMU(deltaPhi),2*M_PI);
  printout(_ns.context->debug_shapes ? ALWAYS : DEBUG, "MyDDCMS",
	   "+   Tubs:     dz=%8.3f [cm] rmin=%8.3f [cm] rmax=%8.3f [cm]"
	   " startPhi=%8.3f [rad] deltaPhi=%8.3f [rad]", dz, rmin, rmax, startPhi, deltaPhi);
  _ns.addSolid(nam, Tube(rmin,rmax,dz,startPhi,deltaPhi));
}
 
/// Converter for <CutTubs/> tags
template <> void Converter<DDLCutTubs>::operator()(xml_h element) const {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam      = e.nameStr();
  double dz       = _ns.attr<double>(e,_CMU(dz));
  double rmin     = _ns.attr<double>(e,_CMU(rMin));
  double rmax     = _ns.attr<double>(e,_CMU(rMax));
  double startPhi = _ns.attr<double>(e,_CMU(startPhi));
  double deltaPhi = _ns.attr<double>(e,_CMU(deltaPhi));
  double lx       = _ns.attr<double>(e,_CMU(lx));
  double ly       = _ns.attr<double>(e,_CMU(ly));
  double lz       = _ns.attr<double>(e,_CMU(lz));
  double tx       = _ns.attr<double>(e,_CMU(tx));
  double ty       = _ns.attr<double>(e,_CMU(ty));
  double tz       = _ns.attr<double>(e,_CMU(tz));
  printout(_ns.context->debug_shapes ? ALWAYS : DEBUG, "MyDDCMS",
	   "+   CutTube:  dz=%8.3f [cm] rmin=%8.3f [cm] rmax=%8.3f [cm]"
	   " startPhi=%8.3f [rad] deltaPhi=%8.3f [rad]...",
	   dz, rmin, rmax, startPhi, deltaPhi);
  _ns.addSolid(nam, CutTube(rmin,rmax,dz,startPhi,deltaPhi,lx,ly,lz,tx,ty,tz));
}

/// Converter for <TruncTubs/> tags
template <> void Converter<DDLTruncTubs>::operator()(xml_h element) const {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam        = e.nameStr();
  double zhalf      = _ns.attr<double>(e,_CMU(zHalf));
  double rmin       = _ns.attr<double>(e,_CMU(rMin));
  double rmax       = _ns.attr<double>(e,_CMU(rMax));
  double startPhi   = _ns.attr<double>(e,_CMU(startPhi));
  double deltaPhi   = _ns.attr<double>(e,_CMU(deltaPhi));
  double cutAtStart = _ns.attr<double>(e,_CMU(cutAtStart));
  double cutAtDelta = _ns.attr<double>(e,_CMU(cutAtDelta));
  bool   cutInside  = _ns.attr<bool>(e,_CMU(cutInside));
  printout(_ns.context->debug_shapes ? ALWAYS : DEBUG, "MyDDCMS",
	   "+   TruncTube:zHalf=%8.3f [cm] rmin=%8.3f [cm] rmax=%8.3f [cm]"
	   " startPhi=%8.3f [rad] deltaPhi=%8.3f [rad] atStart=%8.3f [cm] atDelta=%8.3f [cm] inside:%s",
	   zhalf, rmin, rmax, startPhi, deltaPhi, cutAtStart, cutAtDelta, yes_no(cutInside));
  _ns.addSolid(nam, TruncatedTube(zhalf,rmin,rmax,startPhi,deltaPhi,cutAtStart,cutAtDelta,cutInside));
}

/// Converter for <EllipticalTube/> tags
template <> void Converter<DDLEllipticalTube>::operator()(xml_h element) const   {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam = e.nameStr();
  double dx  = _ns.attr<double>(e,_CMU(xSemiAxis));
  double dy  = _ns.attr<double>(e,_CMU(ySemiAxis));
  double dz  = _ns.attr<double>(e,_CMU(zHeight));
  printout(_ns.context->debug_shapes ? ALWAYS : DEBUG, "MyDDCMS",
	   "+   EllipticalTube xSemiAxis=%8.3f [cm] ySemiAxis=%8.3f [cm] zHeight=%8.3f [cm]",dx,dy,dz);
  _ns.addSolid(nam, EllipticalTube(dx,dy,dz));
}

/// Converter for <Cone/> tags
template <> void Converter<DDLCone>::operator()(xml_h element) const {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam      = e.nameStr();
  double dz       = _ns.attr<double>(e,_CMU(dz));
  double rmin1    = _ns.attr<double>(e,_CMU(rMin1));
  double rmin2    = _ns.attr<double>(e,_CMU(rMin2));
  double rmax1    = _ns.attr<double>(e,_CMU(rMax1));
  double rmax2    = _ns.attr<double>(e,_CMU(rMax2));
  double startPhi = _ns.attr<double>(e,_CMU(startPhi));
  double deltaPhi = _ns.attr<double>(e,_CMU(deltaPhi));
  double phi2     = startPhi + deltaPhi;
  printout(_ns.context->debug_shapes ? ALWAYS : DEBUG, "MyDDCMS",
	   "+   Cone:     dz=%8.3f [cm]"
	   " rmin1=%8.3f [cm] rmax1=%8.3f [cm]"
	   " rmin2=%8.3f [cm] rmax2=%8.3f [cm]"
	   " startPhi=%8.3f [rad] deltaPhi=%8.3f [rad]",
	   dz, rmin1, rmax1, rmin2, rmax2, startPhi, deltaPhi);
  _ns.addSolid(nam, ConeSegment(dz,rmin1,rmax1,rmin2,rmax2,startPhi,phi2));
}

/// Converter for </> tags
template <> void Converter<DDLShapeless>::operator()(xml_h element) const {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam = e.nameStr();
  printout(_ns.context->debug_shapes ? ALWAYS : DEBUG, "MyDDCMS",
	   "+   Shapeless: THIS ONE CAN ONLY BE USED AT THE VOLUME LEVEL -> Assembly%s", nam.c_str());
  _ns.addSolid(nam, Box(1,1,1));
}

/// Converter for <DDLBox/> tags
template <> void Converter<DDLBox>::operator()(xml_h element) const {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string nam = e.nameStr();
  double dx  = _ns.attr<double>(e,_CMU(dx));
  double dy  = _ns.attr<double>(e,_CMU(dy));
  double dz  = _ns.attr<double>(e,_CMU(dz));
  printout(_ns.context->debug_shapes ? ALWAYS : DEBUG, "MyDDCMS",
           "+   Box:      dx=%10.3f [cm] dy=%10.3f [cm] dz=%10.3f [cm]", dx, dy, dz);
  _ns.addSolid(nam, Box(dx,dy,dz));
}

/// DD4hep specific Converter for <Include/> tags: process only the constants
template <> void Converter<include_load>::operator()(xml_h element) const   {
  string fname = element.attr<string>(_U(ref));
  edm::FileInPath fp( fname );
  xml::Document doc;
  doc = xml::DocumentHandler().load( fp.fullPath());
  printout(_param<cms::DDParsingContext>()->debug_includes ? ALWAYS : DEBUG,
           "MyDDCMS","+++ Processing the CMS detector description %s", fname.c_str());
  _option<DDRegistry>()->includes.emplace_back( doc );
}

/// DD4hep specific Converter for <Include/> tags: process only the constants
template <> void Converter<include_unload>::operator()(xml_h element) const   {
  string fname = xml::DocumentHandler::system_path(element);
  xml::DocumentHolder(xml_elt_t(element).document()).assign(nullptr);
  printout(_param<cms::DDParsingContext>()->debug_includes ? ALWAYS : DEBUG,
           "MyDDCMS","+++ Finished processing %s",fname.c_str());
}

/// DD4hep specific Converter for <Include/> tags: process only the constants
template <> void Converter<include_constants>::operator()(xml_h element) const   {
  xml_coll_t(element, _CMU(ConstantsSection)).for_each(Converter<ConstantsSection>(description,param,optional));
}

/// Converter for <Algorithm/> tags
template <> void Converter<DDLAlgorithm>::operator()(xml_h element) const  {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>());
  xml_dim_t e(element);
  string name = e.nameStr();
  if ( _ns.context->disabledAlgs.find(name) != _ns.context->disabledAlgs.end() )   {
    printout(INFO,"MyDDCMS","+++ Skip disabled algorithms: %s",name.c_str());
    return;
  }
  try {
    size_t            idx;
    SensitiveDetector sd;
    string            type = "DDCMS_"+_ns.realName(name);
    while(( idx = type.find( NAMESPACE_SEP )) != string::npos ) type[idx] = '_';

    // SensitiveDetector and Segmentation currently are undefined. Let's keep it like this
    // until we found something better.....
    printout(_ns.context->debug_algorithms ? ALWAYS : DEBUG,
             "MyDDCMS","+++ Start executing algorithm %s....",type.c_str());

    long ret = PluginService::Create<long>(type, &description, _ns.context, &element, &sd);
    if ( ret == 1 )    {
      printout(_ns.context->debug_algorithms ? ALWAYS : DEBUG,
               "MyDDCMS", "+++ Executed algorithm: %08lX = %s", ret, name.c_str());
      return;      
    }
#if 0
    Segmentation      seg;
    DetElement det(PluginService::Create<NamedObject*>(type, &description, _ns.context, &element, &sd));
    if (det.isValid())   {
      // setChildTitles(make_pair(name, det));
      if ( sd.isValid() )   {
        det->flag |= DetElement::Object::HAVE_SENSITIVE_DETECTOR;
      }
      if ( seg.isValid() )   {
        seg->sensitive = sd;
        seg->detector  = det;
      }
    }
    if (!det.isValid())   {
      PluginDebug dbg;
      PluginService::Create<NamedObject*>(type, &description, _ns.context, &element, &sd);
      except("MyDDCMS","Failed to execute subdetector creation plugin. " + dbg.missingFactory(type));
    }
    description.addDetector(det);
#endif
    ///description.addDetector(det);
    printout(ERROR, "MyDDCMS", "++ FAILED  NOT ADDING SUBDETECTOR %08lX = %s",ret, name.c_str());
    return;
  }
  catch (const exception& exc)   {
    printout(ERROR, "MyDDCMS", "++ FAILED    to convert subdetector: %s: %s", name.c_str(), exc.what());
    terminate();
  }
  catch (...)   {
    printout(ERROR, "MyDDCMS", "++ FAILED    to convert subdetector: %s: %s", name.c_str(), "UNKNONW Exception");
    terminate();
  }
}

template <class InputIt, class ForwardIt, class BinOp>
void for_each_token( InputIt first, InputIt last,
		     ForwardIt s_first, ForwardIt s_last,
		     BinOp binary_op)
{
  while( first != last ) {
    const auto pos = std::find_first_of( first, last, s_first, s_last );
    binary_op( first, pos );
    if( pos == last ) break;
    first = std::next( pos );
  }
}

vector<double>
splitNumeric( const string& str, const string& delims = "," )
{
  vector<double> output;

  for_each_token( cbegin( str ), cend( str ),
		  cbegin( delims ), cend( delims ),
		  [&output] ( auto first, auto second ) {
		    if( first != second ) {
		      output.emplace_back(stod(string( first, second )));
		    } 
		  });
  return output;
}

vector<string>
splitString( const string& str, const string& delims = "," )
{
  vector<string> output;

  for_each_token( cbegin( str ), cend( str ),
		  cbegin( delims ), cend( delims ),
		  [&output] ( auto first, auto second ) {
		    if( first != second ) {
		      output.emplace_back( first, second );
		    } 
		  });
  return output;
}

/// Converter for <Vector/> tags
/// FIXME: Check if (parent() == "Algorithm" || parent() == "SpecPar")
template <> void Converter<DDLVector>::operator()( xml_h element ) const {
  DDVectorRegistry registry;

  cms::DDNamespace _ns( _param<cms::DDParsingContext>());
  xml_dim_t e( element );
  string name = e.nameStr();
  string type = _ns.attr<string>(e,_U(type));
  string nEntries = _ns.attr<string>(e,_CMU(nEntries));
  string val = e.text();
  val.erase( remove_if( val.begin(), val.end(), [](unsigned char x){return isspace(x);}), val.end());
  
  printout( _ns.context->debug_constants ? ALWAYS : DEBUG,
	    "MyDDCMS","+++ Vector<%s>:  %s[%s]: %s", type.c_str(), name.c_str(),
	    nEntries.c_str(), val.c_str());

  vector<double> results = splitNumeric( val );
  registry->insert( { name, results } );

  for( auto it : results )
    cout << it << " ";
  cout << "\n";
}

template <> void Converter<debug>::operator()(xml_h dbg) const {
  cms::DDNamespace _ns(_param<cms::DDParsingContext>());
  if ( dbg.hasChild(_CMU(debug_visattr))    ) _ns.context->debug_visattr    = true;
  if ( dbg.hasChild(_CMU(debug_constants))  ) _ns.context->debug_constants  = true;
  if ( dbg.hasChild(_CMU(debug_materials))  ) _ns.context->debug_materials  = true;
  if ( dbg.hasChild(_CMU(debug_rotations))  ) _ns.context->debug_rotations  = true;
  if ( dbg.hasChild(_CMU(debug_shapes))     ) _ns.context->debug_shapes     = true;
  if ( dbg.hasChild(_CMU(debug_volumes))    ) _ns.context->debug_volumes    = true;
  if ( dbg.hasChild(_CMU(debug_placements)) ) _ns.context->debug_placements = true;
  if ( dbg.hasChild(_CMU(debug_namespaces)) ) _ns.context->debug_namespaces = true;
  if ( dbg.hasChild(_CMU(debug_includes))   ) _ns.context->debug_includes   = true;
  if ( dbg.hasChild(_CMU(debug_algorithms)) ) _ns.context->debug_algorithms = true;
}

template <> void Converter<DDRegistry>::operator()(xml_h /* element */) const {
  cms::DDParsingContext* context = _param<cms::DDParsingContext>();
  DDRegistry* res = _option<DDRegistry>();
  cms::DDNamespace       _ns(context);
  int count = 0;

  printout(context->debug_constants ? ALWAYS : DEBUG,
           "MyDDCMS","+++ RESOLVING %ld unknown constants.....",res->unresolvedConst.size());
  while ( !res->unresolvedConst.empty() )   {
    for(auto i=res->unresolvedConst.begin(); i!=res->unresolvedConst.end(); ++i )   {
      const string& n = (*i).first;
      string  rep;
      string& v   = (*i).second;
      size_t idx, idq;
      for(idx=v.find('[',0); idx != string::npos; idx = v.find('[',idx+1) )   {
        idq = v.find(']',idx+1);
        rep = v.substr(idx+1,idq-idx-1);
        auto r = res->allConst.find(rep);
        if ( r != res->allConst.end() )  {
          rep = "("+(*r).second+")";
          v.replace(idx,idq-idx+1,rep);
        }
      }
      if ( v.find(']') == string::npos )  {
        if ( v.find("-+") != string::npos || v.find("+-") != string::npos )   {
          while ( (idx=v.find("-+")) != string::npos )
            v.replace(idx,2,"-");
          while ( (idx=v.find("+-")) != string::npos )
            v.replace(idx,2,"-");
        }
        printout(context->debug_constants ? ALWAYS : DEBUG,
                 "MyDDCMS","+++ [%06ld] ----------  %-40s = %s",
                 res->unresolvedConst.size()-1,n.c_str(),res->originalConst[n].c_str());
        _ns.addConstantNS(n, v, "number");
        res->unresolvedConst.erase(i);
        break;
      }
    }
    if ( ++count > 1000 ) break;
  }
  if ( !res->unresolvedConst.empty() )   {
    for(const auto& e : res->unresolvedConst )
      printout(ERROR,"MyDDCMS","+++ Unresolved constant: %-40s = %s.",e.first.c_str(), e.second.c_str());
    except("MyDDCMS","++ FAILED to resolve %ld constant entries:",res->unresolvedConst.size());
  }
  res->unresolvedConst.clear();
  res->originalConst.clear();
  res->allConst.clear();
}

template <> void Converter<print_xml_doc>::operator()(xml_h element) const {
  string fname = xml::DocumentHandler::system_path(element);
  printout(_param<cms::DDParsingContext>()->debug_includes ? ALWAYS : DEBUG,
           "MyDDCMS","+++ Processing data from: %s",fname.c_str());
}

/// Converter for <DDDefinition/> tags
static long load_dddefinition(Detector& det, xml_h element) {
  static cms::DDParsingContext context(&det);
  cms::DDNamespace _ns(context);
  xml_elt_t dddef(element);
  string fname = xml::DocumentHandler::system_path(element);
  bool open_geometry  = dddef.hasChild(_CMU(open_geometry));
  bool close_geometry = dddef.hasChild(_CMU(close_geometry));

  xml_coll_t(dddef, _U(debug)).for_each(Converter<debug>(det,&context));

  // Here we define the order how XML elements are processed.
  // Be aware of dependencies. This can only defined once.
  // At the end it is a limitation of DOM....
  printout(INFO,"MyDDCMS","+++ Processing the CMS detector description %s",fname.c_str());

  xml::Document doc;
  Converter<print_xml_doc> print_doc(det,&context);
  try  {
    DDRegistry res;
    print_doc((doc=dddef.document()).root());
    xml_coll_t(dddef, _CMU(DisabledAlgo)).for_each(Converter<disabled_algo>(det,&context,&res));
    xml_coll_t(dddef, _CMU(ConstantsSection)).for_each(Converter<ConstantsSection>(det,&context,&res));
    xml_coll_t(dddef, _CMU(VisSection)).for_each(Converter<vissection>(det,&context));
    xml_coll_t(dddef, _CMU(RotationSection)).for_each(Converter<RotationSection>(det,&context));
    xml_coll_t(dddef, _CMU(MaterialSection)).for_each(Converter<MaterialSection>(det,&context));

    xml_coll_t(dddef, _CMU(IncludeSection)).for_each(_CMU(Include), Converter<include_load>(det,&context,&res));

    for(xml::Document d : res.includes )   {
      print_doc((doc=d).root());
      Converter<include_constants>(det,&context,&res)((doc=d).root());
    }
    // Before we continue, we have to resolve all constants NOW!
    Converter<DDRegistry>(det,&context,&res)(dddef);
    // Now we can process the include files one by one.....
    for(xml::Document d : res.includes )   {
      print_doc((doc=d).root());
      xml_coll_t(d.root(),_CMU(MaterialSection)).for_each(Converter<MaterialSection>(det,&context));
    }
    if ( open_geometry )  {
      context.geo_inited = true;
      det.init();
      _ns.addVolume(det.worldVolume());
    }
    for(xml::Document d : res.includes )  {
      print_doc((doc=d).root());
      xml_coll_t(d.root(),_CMU(RotationSection)).for_each(Converter<RotationSection>(det,&context));
    }
    for(xml::Document d : res.includes )  {
      print_doc((doc=d).root());
      xml_coll_t(d.root(), _CMU(SolidSection)).for_each(Converter<SolidSection>(det,&context));
    }
    for(xml::Document d : res.includes )  {
      print_doc((doc=d).root());
      xml_coll_t(d.root(), _CMU(LogicalPartSection)).for_each(Converter<LogicalPartSection>(det,&context));
    }
    for(xml::Document d : res.includes )  {
      print_doc((doc=d).root());
      xml_coll_t(d.root(), _CMU(Algorithm)).for_each(Converter<DDLAlgorithm>(det,&context));
    }
    for(xml::Document d : res.includes )  {
      print_doc((doc=d).root());
      xml_coll_t(d.root(), _CMU(PosPartSection)).for_each(Converter<PosPartSection>(det,&context));
    }

    /// Unload all XML files after processing
    for(xml::Document d : res.includes ) Converter<include_unload>(det,&context,&res)(d.root());

    print_doc((doc=dddef.document()).root());
    // Now process the actual geometry items
    xml_coll_t(dddef, _CMU(SolidSection)).for_each(Converter<SolidSection>(det,&context));
    xml_coll_t(dddef, _CMU(LogicalPartSection)).for_each(Converter<LogicalPartSection>(det,&context));
    xml_coll_t(dddef, _CMU(Algorithm)).for_each(Converter<DDLAlgorithm>(det,&context));
    xml_coll_t(dddef, _CMU(PosPartSection)).for_each(Converter<PosPartSection>(det,&context));
  }
  catch(const exception& e)   {
    printout(ERROR,"MyDDCMS","Exception while processing xml source:%s",doc.uri().c_str());
    printout(ERROR,"MyDDCMS","----> %s", e.what());
    throw;
  }

  /// This should be the end of all processing....close the geometry
  if ( close_geometry )  {
    det.endDocument();
  }
  printout(INFO,"DDDefinition","+++ Finished processing %s",fname.c_str());
  return 1;
}

// Now declare the factory entry for the plugin mechanism
DECLARE_XML_DOC_READER(DDDefinition,load_dddefinition)
