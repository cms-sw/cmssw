/***************************************************************************
                          DDLDivision.cc  -  description
                             -------------------
    begin                : Friday, April 23, 2004
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/

#include "DetectorDescription/Parser/src/DDLDivision.h"
#include "DetectorDescription/Parser/src/DDDividedBox.h"
#include "DetectorDescription/Parser/src/DDDividedTubs.h"
#include "DetectorDescription/Parser/src/DDDividedTrd.h"
#include "DetectorDescription/Parser/src/DDDividedCons.h"
#include "DetectorDescription/Parser/src/DDDividedPolycone.h"
#include "DetectorDescription/Parser/src/DDDividedPolyhedra.h"

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDAxes.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ClhepEvaluator.h"

DDLDivision::DDLDivision( DDLElementRegistry* myreg )
  : DDXMLElement( myreg )
{}

DDLDivision::~DDLDivision( void )
{}

void
DDLDivision::preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{}

void
DDLDivision::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  DCOUT_V('P', "DDLDivision::processElement started");

  DDXMLAttribute atts = getAttributeSet();

  DDName parent = getDDName(nmspace, "parent");
  ClhepEvaluator & ev = myRegistry_->evaluator();
  size_t ax = 0;
  while (DDAxesNames::name(DDAxes(ax)) != atts.find("axis")->second &&
	 DDAxesNames::name(DDAxes(ax)) != "undefined")
    ++ax;

  DDLogicalPart lp(parent);
  if ( !lp.isDefined().second || !lp.solid().isDefined().second ) {
    std::string em("DetectorDescription Parser DDLDivision::processElement(...) failed.");
    em += "  The solid of the parent logical part MUST be defined before the Division is made.";
    em += "\n    name= " + getDDName(nmspace).ns() + ":" + getDDName(nmspace).name() ;
    em += "\n    parent= " + parent.ns() + ":" + parent.name();
    throwError (em);
  }

  DDDivision div;

  if (atts.find("nReplicas") != atts.end() 
      && atts.find("width")  != atts.end()
      && atts.find("offset") != atts.end())
  {
    div = DDDivision(getDDName(nmspace)
		     , parent
		     , DDAxes(ax)
		     , int(ev.eval(nmspace, atts.find("nReplicas")->second))
		     , ev.eval(nmspace, atts.find("width")->second)
		     , ev.eval(nmspace, atts.find("offset")->second));
  }
  else if (atts.find("nReplicas")  != atts.end()
	   && atts.find("offset") != atts.end())
  {
    div = DDDivision(getDDName(nmspace)
		     , parent
		     , DDAxes(ax)
		     , int(ev.eval(nmspace, atts.find("nReplicas")->second))
		     , ev.eval(nmspace, atts.find("offset")->second));
  }
  else if (atts.find("width")     != atts.end()
	   && atts.find("offset") != atts.end())
  {
    DCOUT_V ('D', " width = " << ev.eval(nmspace, atts.find("width")->second) << std::endl);
    DCOUT_V ('D', " offset = " << ev.eval(nmspace, atts.find("offset")->second) << std::endl);
    div = DDDivision(getDDName(nmspace)
		     , parent
		     , DDAxes(ax)
		     , ev.eval(nmspace, atts.find("width")->second)
		     , ev.eval(nmspace, atts.find("offset")->second));
  } else {
    std::string em("DetectorDescription Parser DDLDivision::processElement(...) failed.");
    em += "  Allowed combinations are attributes width&offset OR nReplicas&offset OR nReplicas&width&offset.";
    em += "\n    name= " + getDDName(nmspace).ns() + ":" + getDDName(nmspace).name() ;
    em += "\n    parent= " + parent.ns() + ":" + parent.name();
    throwError (em);
  }
  DDDividedGeometryObject* dg = makeDivider(div, &cpv);
  dg->execute();
  delete dg;

  clear();

  DCOUT_V('P', "DDLDivision::processElement completed");
}

DDDividedGeometryObject*
DDLDivision::makeDivider( const DDDivision& div, DDCompactView* cpv )
{
  DDDividedGeometryObject* dg = NULL;

  switch (div.parent().solid().shape()) 
  {
  case ddbox:
    if (div.axis() == DDAxes::x)
      dg = new DDDividedBoxX(div,cpv);
    else if (div.axis() == DDAxes::y)
      dg = new DDDividedBoxY(div,cpv);
    else if (div.axis() == DDAxes::z)
      dg = new DDDividedBoxZ(div,cpv);
    else {
      std::string s = "DDLDivision can not divide a ";
      s += DDSolidShapesName::name(div.parent().solid().shape());
      s += " along axis " + DDAxesNames::name(div.axis());
      s += ".";
      s += "\n    name= " + div.name().ns() + ":" + div.name().name() ;
      s += "\n    parent= " + div.parent().name().ns() + ":" + div.parent().name().name();
      throwError(s);
    }
    break;

  case ddtubs:
    if (div.axis() == DDAxes::rho)
      dg = new DDDividedTubsRho(div,cpv);
    else if (div.axis() == DDAxes::phi)
      dg = new DDDividedTubsPhi(div,cpv);
    else if (div.axis() == DDAxes::z)
      dg = new DDDividedTubsZ(div,cpv);
    else {
      std::string s = "DDLDivision can not divide a ";
      s += DDSolidShapesName::name(div.parent().solid().shape());
      s += " along axis " + DDAxesNames::name(div.axis());
      s += ".";
      s += "\n    name= " + div.name().ns() + ":" + div.name().name() ;
      s += "\n    parent= " + div.parent().name().ns() + ":" + div.parent().name().name();
      throwError(s);
    }
    break;

  case ddtrap:
    if (div.axis() == DDAxes::x)
      dg = new DDDividedTrdX(div,cpv);
    else if (div.axis() == DDAxes::y )
      dg = new DDDividedTrdY(div,cpv);
    else if (div.axis() == DDAxes::z )
      dg = new DDDividedTrdZ(div,cpv);
    else {
      std::string s = "DDLDivision can not divide a ";
      s += DDSolidShapesName::name(div.parent().solid().shape());
      s += " along axis ";
      s += DDAxesNames::name(div.axis());
      s += ".";
      s += "\n    name= " + div.name().ns() + ":" + div.name().name() ;
      s += "\n    parent= " + div.parent().name().ns() + ":" + div.parent().name().name();
      throwError(s);
    }
    break;

  case ddcons:
    if (div.axis() == DDAxes::rho)
      dg = new DDDividedConsRho(div,cpv);
    else if (div.axis() == DDAxes::phi)
      dg = new DDDividedConsPhi(div,cpv);
    else if (div.axis() == DDAxes::z)
      dg = new DDDividedConsZ(div,cpv);
    else {
      std::string s = "DDLDivision can not divide a ";
      s += DDSolidShapesName::name(div.parent().solid().shape());
      s += " along axis " + DDAxesNames::name(div.axis());
      s += ".";
      s += "\n    name= " + div.name().ns() + ":" + div.name().name() ;
      s += "\n    parent= " + div.parent().name().ns() + ":" + div.parent().name().name();
      throwError(s);
    }
    break;

  case ddpolycone_rrz:
    if (div.axis() == DDAxes::rho)
      dg = new DDDividedPolyconeRho(div,cpv);
    else if (div.axis() == DDAxes::phi)
      dg = new DDDividedPolyconePhi(div,cpv);
    else if (div.axis() == DDAxes::z)
      dg = new DDDividedPolyconeZ(div,cpv);
    else {
      std::string s = "DDLDivision can not divide a ";
      s += DDSolidShapesName::name(div.parent().solid().shape());
      s += " along axis ";
      s += DDAxesNames::name(div.axis());
      s += ".";
      s += "\n    name= " + div.name().ns() + ":" + div.name().name() ;
      s += "\n    parent= " + div.parent().name().ns() + ":" + div.parent().name().name();
      throwError(s);
    }
    break;

  case ddpolyhedra_rrz:
    if (div.axis() == DDAxes::rho)
      dg = new DDDividedPolyhedraRho(div,cpv);
    else if (div.axis() == DDAxes::phi)
      dg = new DDDividedPolyhedraPhi(div,cpv);
    else if (div.axis() == DDAxes::z)
      dg = new DDDividedPolyhedraZ(div,cpv);
    else {
      std::string s = "DDLDivision can not divide a ";
      s += DDSolidShapesName::name(div.parent().solid().shape());
      s += " along axis ";
      s += DDAxesNames::name(div.axis());
      s += ".";
      s += "\n    name= " + div.name().ns() + ":" + div.name().name() ;
      s += "\n    parent= " + div.parent().name().ns() + ":" + div.parent().name().name();
      throwError(s);
    }
    break;

  case ddpolycone_rz:
  case ddpolyhedra_rz: {
    std::string s = "ERROR:  A Polycone or Polyhedra can not be divided on any axis if it's\n";
    s += "original definition used r and z instead of ZSections. This has\n";
    s += "not (yet) been implemented.";
    s += "\n    name= " + div.name().ns() + ":" + div.name().name() ;
    s += "\n    parent= " + div.parent().name().ns() + ":" + div.parent().name().name();
  }
    break;

  case ddunion: 
  case ddsubtraction: 
  case ddintersection: 
  case ddreflected: 
  case ddshapeless: 
  case ddpseudotrap: 
  case ddtrunctubs:
  case dd_not_init: {
    std::string s = "DDLDivision can not divide a ";
    s += DDSolidShapesName::name(div.parent().solid().shape());
    s += " at all (yet?).  Requested axis was ";
    s += DDAxesNames::name(div.axis());
    s += ".\n";
    throwError(s);
  }
    break;
  default:
    break;
  }
  return dg;

}
