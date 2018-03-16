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
#include "DetectorDescription/DDCMS/interface/DDCMS.h"

#include "TSystem.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"

#include <climits>
#include <iostream>
#include <iomanip>
#include <set>
#include <map>

using namespace std;
using namespace dd4hep;
using namespace dd4hep::cms;

namespace dd4hep {

  namespace {

    static UInt_t unique_mat_id = 0xAFFEFEED;


    class disabled_algo;
    class include_constants;
    class include_load;
    class include_unload;
    class print_xml_doc;
    class constantssection;
    class constant;
    class resolve   {
    public:
      std::vector<xml::Document> includes;
      std::map<std::string,std::string>  unresolvedConst, allConst, originalConst;
    };

    class materialsection;
    class elementarymaterial;
    class compositematerial;
  
    class rotationsection;
    class rotation;
    class transform3d;

    class pospartsection;
    class pospart;

    class logicalpartsection;
    class logicalpart;

    class solidsection;
    class trapezoid;
    class polycone;
    class torus;
    class tubs;
    class box;
    class unionsolid;
    class intersectionsolid;
    class polyhedra;
    class subtractionsolid;
    
    class algorithm;    

    class vissection;
    class vis;
    class debug;
  }

  /// Converter instances implemented in this compilation unit
  template <> void Converter<debug>::operator()(xml_h element) const;
  template <> void Converter<print_xml_doc>::operator()(xml_h element) const;
  template <> void Converter<disabled_algo>::operator()(xml_h element) const;
  
  /// Converter for <ConstantsSection/> tags
  template <> void Converter<constantssection>::operator()(xml_h element) const;
  template <> void Converter<constant>::operator()(xml_h element) const;
  template <> void Converter<resolve>::operator()(xml_h element) const;

  /// Converter for <VisSection/> tags
  template <> void Converter<vissection>::operator()(xml_h element) const;
  /// Convert compact visualization attributes
  template <> void Converter<vis>::operator()(xml_h element) const;

  /// Converter for <MaterialSection/> tags
  template <> void Converter<materialsection>::operator()(xml_h element) const;
  template <> void Converter<elementarymaterial>::operator()(xml_h element) const;
  template <> void Converter<compositematerial>::operator()(xml_h element) const;

  /// Converter for <RotationSection/> tags
  template <> void Converter<rotationsection>::operator()(xml_h element) const;
  /// Converter for <Rotation/> tags
  template <> void Converter<rotation>::operator()(xml_h element) const;
  template <> void Converter<transform3d>::operator()(xml_h element) const;

  /// Generic converter for  <LogicalPartSection/> tags
  template <> void Converter<logicalpartsection>::operator()(xml_h element) const;
  template <> void Converter<logicalpart>::operator()(xml_h element) const;

  template <> void Converter<pospartsection>::operator()(xml_h element) const;
  /// Converter for <PosPart/> tags
  template <> void Converter<pospart>::operator()(xml_h element) const;

  /// Generic converter for solids: <SolidSection/> tags
  template <> void Converter<solidsection>::operator()(xml_h element) const;
  /// Converter for <UnionSolid/> tags
  template <> void Converter<unionsolid>::operator()(xml_h element) const;
  /// Converter for <SubtractionSolid/> tags
  template <> void Converter<subtractionsolid>::operator()(xml_h element) const;
  /// Converter for <IntersectionSolid/> tags
  template <> void Converter<intersectionsolid>::operator()(xml_h element) const;
  /// Converter for <Trapezoid/> tags
  template <> void Converter<trapezoid>::operator()(xml_h element) const;
  /// Converter for <Polycone/> tags
  template <> void Converter<polycone>::operator()(xml_h element) const;
  /// Converter for <Polyhedra/> tags
  template <> void Converter<polyhedra>::operator()(xml_h element) const;
  /// Converter for <Torus/> tags
  template <> void Converter<torus>::operator()(xml_h element) const;
  /// Converter for <Tubs/> tags
  template <> void Converter<tubs>::operator()(xml_h element) const;
  /// Converter for <Box/> tags
  template <> void Converter<box>::operator()(xml_h element) const;

  /// Converter for <Algorithm/> tags
  template <> void Converter<algorithm>::operator()(xml_h element) const;

  /// DD4hep specific: Load include file
  template <> void Converter<include_load>::operator()(xml_h element) const;
  /// DD4hep specific: Unload include file
  template <> void Converter<include_unload>::operator()(xml_h element) const;
  /// DD4hep specific: Process constants objects
  template <> void Converter<include_constants>::operator()(xml_h element) const;
}

/// Converter for <ConstantsSection/> tags
template <> void Converter<constantssection>::operator()(xml_h element) const  {
  Namespace _ns(_param<ParsingContext>(), element);
  xml_coll_t(element, _CMU(Constant)).for_each(Converter<constant>(description,_ns.context,optional));
}

/// Converter for <VisSection/> tags
template <> void Converter<vissection>::operator()(xml_h element) const  {
  Namespace _ns(_param<ParsingContext>(), element);
  xml_coll_t(element, _CMU(vis)).for_each(Converter<vis>(description,_ns.context,optional));
}

/// Converter for <MaterialSection/> tags
template <> void Converter<materialsection>::operator()(xml_h element) const   {
  Namespace _ns(_param<ParsingContext>(), element);
  xml_coll_t(element, _CMU(ElementaryMaterial)).for_each(Converter<elementarymaterial>(description,_ns.context,optional));
  xml_coll_t(element, _CMU(CompositeMaterial)).for_each(Converter<compositematerial>(description,_ns.context,optional));
}

template <> void Converter<rotationsection>::operator()(xml_h element) const   {
  Namespace _ns(_param<ParsingContext>(), element);
  xml_coll_t(element, _CMU(Rotation)).for_each(Converter<rotation>(description,_ns.context,optional));
}

template <> void Converter<pospartsection>::operator()(xml_h element) const   {
  Namespace _ns(_param<ParsingContext>(), element);
  xml_coll_t(element, _CMU(PosPart)).for_each(Converter<pospart>(description,_ns.context,optional));
}

/// Generic converter for  <LogicalPartSection/> tags
template <> void Converter<logicalpartsection>::operator()(xml_h element) const   {
  Namespace _ns(_param<ParsingContext>(), element);
  xml_coll_t(element, _CMU(LogicalPart)).for_each(Converter<logicalpart>(description,_ns.context,optional));
}

template <> void Converter<disabled_algo>::operator()(xml_h element) const   {
  ParsingContext* c = _param<ParsingContext>();
  c->disabledAlgs.insert(element.attr<string>(_U(name)));
}

/// Generic converter for  <SolidSection/> tags
template <> void Converter<solidsection>::operator()(xml_h element) const   {
  Namespace _ns(_param<ParsingContext>(), element);
  for(xml_coll_t solid(element, _U(star)); solid; ++solid)   {
    string tag = solid.tag();
    if ( tag == "Box" )
      Converter<box>(description,_ns.context,optional)(solid);
    else if ( tag == "Polycone" )
      Converter<polycone>(description,_ns.context,optional)(solid);
    else if ( tag == "Tubs" )
      Converter<tubs>(description,_ns.context,optional)(solid);
    else if ( tag == "Torus" )
      Converter<torus>(description,_ns.context,optional)(solid);
    else if ( tag == "Trapezoid" )
      Converter<trapezoid>(description,_ns.context,optional)(solid);
    else if ( tag == "UnionSolid" )
      Converter<unionsolid>(description,_ns.context,optional)(solid);
    else if ( tag == "SubtractionSolid" )
      Converter<subtractionsolid>(description,_ns.context,optional)(solid);
    else if ( tag == "IntersectionSolid" )
      Converter<intersectionsolid>(description,_ns.context,optional)(solid);
    else if ( tag == "Polyhedra" ) {
      Converter<polyhedra>(description,_ns.context,optional)(solid);
      std::cout << "Polyhedra is not implemented yet!\n";
    }
    else  {
      string nam = xml_dim_t(solid).nameStr();
      printout(ERROR,"MyDDCMS","+++ Request to process unknown shape %s [%s]",
               nam.c_str(), tag.c_str());
    }
  }
}

/// Converter for <Constant/> tags
template <> void Converter<constant>::operator()(xml_h element) const  {
  Namespace _ns(_param<ParsingContext>());
  resolve*  res  = _option<resolve>();
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
      val.insert(idx,_ns.name);
    else if ( idp != string::npos && idp < idq )
      val[idp] = '_';
    idx = val.find('[',idx);
  }
  while ( (idx=val.find(':')) != string::npos ) val[idx]='_';
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
  Namespace _ns(_param<ParsingContext>());
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

/// Converter for <ElementaryMaterial/> tags
template <> void Converter<elementarymaterial>::operator()(xml_h element) const   {
  Namespace     _ns(_param<ParsingContext>());
  xml_dim_t     xmat(element);
  string        nam = _ns.prepend(xmat.nameStr());
  TGeoManager&  mgr = description.manager();
  TGeoMaterial* mat = mgr.GetMaterial(nam.c_str());
  if ( 0 == mat )   {
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
    if (0 == medium) {
      --unique_mat_id;
      medium = new TGeoMedium(matname, unique_mat_id, mix);
      medium->SetTitle("material");
      medium->SetUniqueID(unique_mat_id);
    }
  }
}

/// Converter for <CompositeMaterial/> tags
template <> void Converter<compositematerial>::operator()(xml_h element) const   {
  Namespace     _ns(_param<ParsingContext>());
  xml_dim_t     xmat(element);
  string        nam = _ns.prepend(xmat.nameStr());
  TGeoManager&  mgr = description.manager();
  TGeoMaterial* mat = mgr.GetMaterial(nam.c_str());
  if ( 0 == mat )   {
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
      string    fracname = _ns.real_name(xfrac_mat.nameStr());

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
    if (0 == medium) {
      --unique_mat_id;
      medium = new TGeoMedium(matname, unique_mat_id, mix);
      medium->SetTitle("material");
      medium->SetUniqueID(unique_mat_id);
    }
    
  }
}

/// Converter for <Rotation/> tags
template <> void Converter<rotation>::operator()(xml_h element) const  {
  ParsingContext* ctx = _param<ParsingContext>();
  Namespace _ns(ctx);
  xml_dim_t xrot(element);
  string    nam    = xrot.nameStr();
  double    thetaX = xrot.hasAttr(_CMU(thetaX)) ? _ns.attr<double>(xrot,_CMU(thetaX)) : 0e0;
  double    phiX   = xrot.hasAttr(_CMU(phiX))   ? _ns.attr<double>(xrot,_CMU(phiX))   : 0e0;
  double    thetaY = xrot.hasAttr(_CMU(thetaY)) ? _ns.attr<double>(xrot,_CMU(thetaY)) : 0e0;
  double    phiY   = xrot.hasAttr(_CMU(phiY))   ? _ns.attr<double>(xrot,_CMU(phiY))   : 0e0;
  double    thetaZ = xrot.hasAttr(_CMU(thetaZ)) ? _ns.attr<double>(xrot,_CMU(thetaZ)) : 0e0;
  double    phiZ   = xrot.hasAttr(_CMU(phiZ))   ? _ns.attr<double>(xrot,_CMU(phiZ))   : 0e0;
  Rotation3D rot = make_rotation3D(thetaX, phiX, thetaY, phiY, thetaZ, phiZ);
  printout(ctx->debug_rotations ? ALWAYS : DEBUG,
           "MyDDCMS","+++ Adding rotation: %-32s: (theta/phi)[rad] X: %6.3f %6.3f Y: %6.3f %6.3f Z: %6.3f %6.3f",
           _ns.prepend(nam).c_str(),thetaX,phiX,thetaY,phiY,thetaZ,phiZ);
  _ns.addRotation(nam, rot);
}

/// Converter for <Logicalpart/> tags
template <> void Converter<logicalpart>::operator()(xml_h element) const {
  Namespace _ns(_param<ParsingContext>());
  xml_dim_t e(element);
  string    sol = e.child(_CMU(rSolid)).attr<string>(_U(name));
  string    mat = e.child(_CMU(rMaterial)).attr<string>(_U(name));
  _ns.addVolume(Volume(e.nameStr(), _ns.solid(sol), _ns.material(mat)));
}

/// Helper converter
template <> void Converter<transform3d>::operator()(xml_h element) const {
  Namespace    _ns(_param<ParsingContext>());
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
template <> void Converter<pospart>::operator()(xml_h element) const {
  Namespace _ns(_param<ParsingContext>());
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
    Converter<transform3d>(description,param,&trafo)(element);
    pv = parent.placeVolume(child,copy,trafo);
  }
  if ( !pv.isValid() )   {
    printout(ERROR,"MyDDCMS","+++ Placement FAILED! Parent:%s Child:%s Valid:%s",
             parent.name(), child_nam.c_str(), yes_no(child.isValid()));
  }
}

template <typename TYPE>
static void convert_boolean(ParsingContext* ctx, xml_h element)   {
  Namespace   _ns(ctx);
  xml_dim_t   e(element);
  string      nam = e.nameStr();
  Solid       solids[2];
  Solid       boolean;
  int cnt=0;

  for(xml_coll_t c(element, _CMU(rSolid)); cnt<2 && c; ++c, ++cnt)
    solids[cnt] = _ns.solid(c.attr<string>(_U(name)));

  if ( cnt != 2 )   {
    except("MyDDCMS","+++ Failed to create blooean solid %s. Found only %d parts.",nam.c_str(), cnt);
  }
  printout(_ns.context->debug_placements ? ALWAYS : DEBUG, "MyDDCMS",
           "+++ SubtractionSolid: %s Left: %-32s Right: %-32s",
           nam.c_str(), solids[0]->GetName(), solids[1]->GetName());

  if ( solids[0].isValid() && solids[1].isValid() )  {
    Transform3D trafo;
    Converter<transform3d>(*ctx->description,ctx,&trafo)(element);
    boolean = TYPE(solids[0],solids[1],trafo);
  }
  if ( !boolean.isValid() )
    except("MyDDCMS","+++ FAILED to construct subtraction solid: %s",nam.c_str());
  _ns.addSolid(nam,boolean);
}

/// Converter for <SubtractionSolid/> tags
template <> void Converter<unionsolid>::operator()(xml_h element) const   {
  convert_boolean<UnionSolid>(_param<ParsingContext>(),element);
}

/// Converter for <SubtractionSolid/> tags
template <> void Converter<subtractionsolid>::operator()(xml_h element) const   {
  convert_boolean<SubtractionSolid>(_param<ParsingContext>(),element);
}

/// Converter for <SubtractionSolid/> tags
template <> void Converter<intersectionsolid>::operator()(xml_h element) const   {
  convert_boolean<IntersectionSolid>(_param<ParsingContext>(),element);
}

/// Converter for <Polycone/> tags
template <> void Converter<polycone>::operator()(xml_h element) const {
  Namespace _ns(_param<ParsingContext>());
  xml_dim_t e(element);
  string nam      = e.nameStr();
  double startPhi = _ns.attr<double>(e,_CMU(startPhi));
  double deltaPhi = _ns.attr<double>(e,_CMU(deltaPhi));
  vector<double> z, rmin, rmax;
  
  for(xml_coll_t zplane(element, _CMU(ZSection)); zplane; ++zplane)   {
    rmin.push_back(_ns.attr<double>(zplane,_CMU(rMin)));
    rmax.push_back(_ns.attr<double>(zplane,_CMU(rMax)));
    z.push_back(_ns.attr<double>(zplane,_CMU(z)));
  }
  printout(_ns.context->debug_shapes ? ALWAYS : DEBUG, "MyDDCMS",
           "+   Polycone: startPhi=%10.3f [rad] deltaPhi=%10.3f [rad]  %4ld z-planes",
           startPhi, deltaPhi, z.size());
  _ns.addSolid(nam, Polycone(startPhi,deltaPhi,rmin,rmax,z));
}

/// Converter for <Polyhedra/> tags
template <> void Converter<polyhedra>::operator()(xml_h element) const {
  Namespace _ns(_param<ParsingContext>());
  xml_dim_t e(element);
  string nam      = e.nameStr();
  int numSide = _ns.attr<int>(e,_CMU(numSide));
  double startPhi = _ns.attr<double>(e,_CMU(startPhi));
  double deltaPhi = _ns.attr<double>(e,_CMU(deltaPhi));
  vector<double> z, rmin, rmax;
  
  for(xml_coll_t zplane(element, _CMU(ZSection)); zplane; ++zplane)   {
    rmin.push_back(_ns.attr<double>(zplane,_CMU(rMin)));
    rmax.push_back(_ns.attr<double>(zplane,_CMU(rMax)));
    z.push_back(_ns.attr<double>(zplane,_CMU(z)));
  }
  printout(_ns.context->debug_shapes ? ALWAYS : DEBUG, "MyDDCMS",
           "+   Polyhedra: numSide=%d startPhi=%10.3f [rad] deltaPhi=%10.3f [rad]  %4ld z-planes",
           numSide, startPhi, deltaPhi, z.size());
  //_ns.addSolid(nam, PolyhedraRegular(numSide,startPhi,deltaPhi,rmin,rmax,z));
    _ns.addSolid(nam, Polycone(startPhi,deltaPhi,rmin,rmax,z));
}

/// Converter for <Torus/> tags
template <> void Converter<torus>::operator()(xml_h element) const   {
  Namespace _ns(_param<ParsingContext>());
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

/// Converter for <Trapezoid/> tags
template <> void Converter<trapezoid>::operator()(xml_h element) const {
  Namespace _ns(_param<ParsingContext>());
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
  double phi      = _ns.attr<double>(e,_U(phi));
  double theta    = _ns.attr<double>(e,_U(theta));
  printout(_ns.context->debug_shapes ? ALWAYS : DEBUG, "MyDDCMS",
           "+   Trapezoid:  dz=%10.3f [cm] alp1:%.3f bl1=%.3f tl1=%.3f alp2=%.3f bl2=%.3f tl2=%.3f h2=%.3f phi=%.3f theta=%.3f",
           dz, alp1, bl1, tl1, h1, alp2, bl2, tl2, h2, phi, theta);
  _ns.addSolid(nam, Trap(dz, theta, phi, h1, bl1, tl1, alp1, h2, bl2, tl2, alp2));
}

/// Converter for <Tubs/> tags
template <> void Converter<tubs>::operator()(xml_h element) const {
  Namespace _ns(_param<ParsingContext>());
  xml_dim_t e(element);
  string nam      = e.nameStr();
  double dz       = _ns.attr<double>(e,_CMU(dz));
  double rmin     = _ns.attr<double>(e,_CMU(rMin));
  double rmax     = _ns.attr<double>(e,_CMU(rMax));
  double startPhi = _ns.attr<double>(e,_CMU(startPhi));
  double deltaPhi = _ns.attr<double>(e,_CMU(deltaPhi));
  printout(_ns.context->debug_shapes ? ALWAYS : DEBUG, "MyDDCMS",
           "+   Tubs:     dz=%10.3f [cm] rmin=%10.3f [cm] rmax=%10.3f [cm]"
           " startPhi=%10.3f [rad] deltaPhi=%10.3f [rad]", dz, rmin, rmax, startPhi, deltaPhi);
  _ns.addSolid(nam, Tube(rmin,rmax,dz,startPhi,deltaPhi));
}

/// Converter for <Box/> tags
template <> void Converter<box>::operator()(xml_h element) const {
  Namespace _ns(_param<ParsingContext>());
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
  TString fname = element.attr<string>(_U(ref)).c_str();
  const char* path = gSystem->Getenv("DDCMS_XML_PATH");
  xml::Document doc;
  if ( path && gSystem->FindFile(path,fname) )
    doc = xml::DocumentHandler().load(fname.Data());
  else
    doc = xml::DocumentHandler().load(element, element.attr_value(_U(ref)));
  fname = xml::DocumentHandler::system_path(doc.root());
  printout(_param<ParsingContext>()->debug_includes ? ALWAYS : DEBUG,
           "MyDDCMS","+++ Processing the CMS detector description %s",fname.Data());
  _option<resolve>()->includes.push_back(doc);
}

/// DD4hep specific Converter for <Include/> tags: process only the constants
template <> void Converter<include_unload>::operator()(xml_h element) const   {
  string fname = xml::DocumentHandler::system_path(element);
  xml::DocumentHolder(xml_elt_t(element).document()).assign(0);
  printout(_param<ParsingContext>()->debug_includes ? ALWAYS : DEBUG,
           "MyDDCMS","+++ Finished processing %s",fname.c_str());
}

/// DD4hep specific Converter for <Include/> tags: process only the constants
template <> void Converter<include_constants>::operator()(xml_h element) const   {
  xml_coll_t(element, _CMU(ConstantsSection)).for_each(Converter<constantssection>(description,param,optional));
}

/// Converter for <Algorithm/> tags
template <> void Converter<algorithm>::operator()(xml_h element) const  {
  Namespace _ns(_param<ParsingContext>());
  xml_dim_t e(element);
  string name = e.nameStr();
  if ( _ns.context->disabledAlgs.find(name) != _ns.context->disabledAlgs.end() )   {
    printout(INFO,"MyDDCMS","+++ Skip disabled algorithms: %s",name.c_str());
    return;
  }
  try {
    SensitiveDetector sd;
    Segmentation      seg;
    string            type = "DDCMS_"+_ns.real_name(name);

    // SensitiveDetector and Segmentation currently are undefined. Let's keep it like this
    // until we found something better.....
    printout(_ns.context->debug_algorithms ? ALWAYS : DEBUG,
             "MyDDCMS","+++ Start executing algorithm %s....",type.c_str());
    LogDebug context(e.nameStr(),true);
    long ret = PluginService::Create<long>(type, &description, _ns.context, &element, &sd);
    if ( ret == 1 )    {
      printout(_ns.context->debug_algorithms ? ALWAYS : DEBUG,
               "MyDDCMS", "+++ Executed algorithm: %08lX = %s", ret, name.c_str());
      return;      
    }
#if 0
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

template <> void Converter<debug>::operator()(xml_h dbg) const {
  Namespace _ns(_param<ParsingContext>());
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
  LogDebug::setDebugAlgorithms(_ns.context->debug_algorithms);
}

template <> void Converter<resolve>::operator()(xml_h /* element */) const {
  ParsingContext* ctx = _param<ParsingContext>();
  resolve*        res = _option<resolve>();
  Namespace       _ns(ctx);
  int count = 0;

  printout(ctx->debug_constants ? ALWAYS : DEBUG,
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
        printout(ctx->debug_constants ? ALWAYS : DEBUG,
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
  printout(_param<ParsingContext>()->debug_includes ? ALWAYS : DEBUG,
           "MyDDCMS","+++ Processing data from: %s",fname.c_str());
}

/// Converter for <DDDefinition/> tags
static long load_dddefinition(Detector& det, xml_h element) {
  static ParsingContext ctxt(&det);
  Namespace _ns(ctxt);
  xml_elt_t dddef(element);
  string fname = xml::DocumentHandler::system_path(element);
  bool open_geometry  = dddef.hasChild(_CMU(open_geometry));
  bool close_geometry = dddef.hasChild(_CMU(close_geometry));

  xml_coll_t(dddef, _U(debug)).for_each(Converter<debug>(det,&ctxt));

  // Here we define the order how XML elements are processed.
  // Be aware of dependencies. This can only defined once.
  // At the end it is a limitation of DOM....
  printout(INFO,"MyDDCMS","+++ Processing the CMS detector description %s",fname.c_str());

  xml::Document doc;
  Converter<print_xml_doc> print_doc(det,&ctxt);
  try  {
    resolve res;
    print_doc((doc=dddef.document()).root());
    xml_coll_t(dddef, _CMU(DisabledAlgo)).for_each(Converter<disabled_algo>(det,&ctxt,&res));
    xml_coll_t(dddef, _CMU(ConstantsSection)).for_each(Converter<constantssection>(det,&ctxt,&res));
    xml_coll_t(dddef, _CMU(VisSection)).for_each(Converter<vissection>(det,&ctxt));
    xml_coll_t(dddef, _CMU(RotationSection)).for_each(Converter<rotationsection>(det,&ctxt));
    xml_coll_t(dddef, _CMU(MaterialSection)).for_each(Converter<materialsection>(det,&ctxt));

    xml_coll_t(dddef, _CMU(IncludeSection)).for_each(_CMU(Include), Converter<include_load>(det,&ctxt,&res));

    for(xml::Document d : res.includes )   {
      print_doc((doc=d).root());
      Converter<include_constants>(det,&ctxt,&res)((doc=d).root());
    }
    // Before we continue, we have to resolve all constants NOW!
    Converter<resolve>(det,&ctxt,&res)(dddef);
    // Now we can process the include files one by one.....
    for(xml::Document d : res.includes )   {
      print_doc((doc=d).root());
      xml_coll_t(d.root(),_CMU(MaterialSection)).for_each(Converter<materialsection>(det,&ctxt));
    }
    if ( open_geometry )  {
      ctxt.geo_inited = true;
      det.init();
      _ns.addVolume(det.worldVolume());
    }
    for(xml::Document d : res.includes )  {
      print_doc((doc=d).root());
      xml_coll_t(d.root(),_CMU(RotationSection)).for_each(Converter<rotationsection>(det,&ctxt));
    }
    for(xml::Document d : res.includes )  {
      print_doc((doc=d).root());
      xml_coll_t(d.root(), _CMU(SolidSection)).for_each(Converter<solidsection>(det,&ctxt));
    }
    for(xml::Document d : res.includes )  {
      print_doc((doc=d).root());
      xml_coll_t(d.root(), _CMU(LogicalPartSection)).for_each(Converter<logicalpartsection>(det,&ctxt));
    }
    for(xml::Document d : res.includes )  {
      print_doc((doc=d).root());
      xml_coll_t(d.root(), _CMU(Algorithm)).for_each(Converter<algorithm>(det,&ctxt));
    }
    for(xml::Document d : res.includes )  {
      print_doc((doc=d).root());
      xml_coll_t(d.root(), _CMU(PosPartSection)).for_each(Converter<pospartsection>(det,&ctxt));
    }

    /// Unload all XML files after processing
    for(xml::Document d : res.includes ) Converter<include_unload>(det,&ctxt,&res)(d.root());

    print_doc((doc=dddef.document()).root());
    // Now process the actual geometry items
    xml_coll_t(dddef, _CMU(SolidSection)).for_each(Converter<solidsection>(det,&ctxt));
    xml_coll_t(dddef, _CMU(LogicalPartSection)).for_each(Converter<logicalpartsection>(det,&ctxt));
    xml_coll_t(dddef, _CMU(Algorithm)).for_each(Converter<algorithm>(det,&ctxt));
    xml_coll_t(dddef, _CMU(PosPartSection)).for_each(Converter<pospartsection>(det,&ctxt));

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
