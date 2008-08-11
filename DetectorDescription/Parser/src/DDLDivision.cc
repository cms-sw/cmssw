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



#include "DDLElementRegistry.h"
#include "DDLDivision.h"
#include "DDXMLElement.h"
#include "DDDividedBox.h"
#include "DDDividedTubs.h"
#include "DDDividedTrd.h"
#include "DDDividedCons.h"
#include "DDDividedPolycone.h"
#include "DDDividedPolyhedra.h"
#include "DDDividedGeometryObject.h"

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDAxes.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/DDDivision.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Base/interface/DDException.h"

#include "CLHEP/Units/SystemOfUnits.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

#include <string>

DDLDivision::DDLDivision()
{
}

DDLDivision::~DDLDivision()
{
}


void DDLDivision::preProcessElement (const std::string& type, const std::string& nmspace)
{

}

void DDLDivision::processElement (const std::string& type, const std::string& nmspace)
{
  DCOUT_V('P', "DDLDivision::processElement started");

  DDXMLAttribute atts = getAttributeSet();

  DDName parent = getDDName(nmspace, "parent");
  ExprEvalInterface & ev = ExprEvalSingleton::instance();
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
  DDDividedGeometryObject* dg = makeDivider(div);
  dg->execute();
  delete dg;

  clear();

  DCOUT_V('P', "DDLDivision::processElement completed");
}

DDDividedGeometryObject* DDLDivision::makeDivider(const DDDivision & div)
{
  DDDividedGeometryObject* dg = NULL;

  switch (div.parent().solid().shape()) 
    {
    case ddbox:
      if (div.axis() == x)
	dg = new DDDividedBoxX(div);
      else if (div.axis() == y)
	dg = new DDDividedBoxY(div);
      else if (div.axis() == z)
	dg = new DDDividedBoxZ(div);
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
      if (div.axis() == rho)
	dg = new DDDividedTubsRho(div);
      else if (div.axis() == phi)
	dg = new DDDividedTubsPhi(div);
      else if (div.axis() == z)
	dg = new DDDividedTubsZ(div);
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
      if (div.axis() == x)
	dg = new DDDividedTrdX(div);
      else if (div.axis() == y )
	dg = new DDDividedTrdY(div);
      else if (div.axis() == z )
	dg = new DDDividedTrdZ(div);
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
      if (div.axis() == rho)
	dg = new DDDividedConsRho(div);
      else if (div.axis() == phi)
	dg = new DDDividedConsPhi(div);
      else if (div.axis() == z)
	dg = new DDDividedConsZ(div);
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
      if (div.axis() == rho)
	dg = new DDDividedPolyconeRho(div);
      else if (div.axis() == phi)
	dg = new DDDividedPolyconePhi(div);
      else if (div.axis() == z)
	dg = new DDDividedPolyconeZ(div);
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
      if (div.axis() == rho)
	dg = new DDDividedPolyhedraRho(div);
      else if (div.axis() == phi)
	dg = new DDDividedPolyhedraPhi(div);
      else if (div.axis() == z)
	dg = new DDDividedPolyhedraZ(div);
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
