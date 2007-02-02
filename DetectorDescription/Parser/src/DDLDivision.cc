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



#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/interface/DDLDivision.h"
#include "DetectorDescription/Parser/interface/DDXMLElement.h"
#include "DetectorDescription/Parser/interface/DDDividedBox.h"
#include "DetectorDescription/Parser/interface/DDDividedTubs.h"
#include "DetectorDescription/Parser/interface/DDDividedTrd.h"
#include "DetectorDescription/Parser/interface/DDDividedCons.h"
#include "DetectorDescription/Parser/interface/DDDividedPolycone.h"
#include "DetectorDescription/Parser/interface/DDDividedPolyhedra.h"
#include "DetectorDescription/Parser/interface/DDDividedGeometryObject.h"

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

  DDDivision div;
  try {
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
      }
    DDDividedGeometryObject* dg = makeDivider(div);
    dg->execute();
    delete dg; // i think :-)
  } catch (DDException& e) {
    std::string msg = e.what();
    msg += "\nDDLDivision failed to create DDDivision.\n";
    msg += "\n\tname: " + atts.find("name")->second;
    throwError(msg);
  }

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
      break;

    case ddtubs:
      if (div.axis() == rho)
	dg = new DDDividedTubsRho(div);
      else if (div.axis() == phi)
	dg = new DDDividedTubsPhi(div);
      if (div.axis() == z)
	dg = new DDDividedTubsZ(div);
      break;

    case ddtrap:
      if (div.axis() == x)
	dg = new DDDividedTrdX(div);
      else if (div.axis() == y )
	dg = new DDDividedTrdY(div);
      else if (div.axis() == z )
	dg = new DDDividedTrdZ(div);
      else {
	std::string s = "DDDividedAlgorithm can not divide a trap or trd on axis ";
	s += DDAxesNames::name(div.axis());
	s += ".\n";
	throw DDException(s);
      }
      break;

    case ddcons:
      if (div.axis() == rho)
	dg = new DDDividedConsRho(div);
      else if (div.axis() == phi)
	dg = new DDDividedConsPhi(div);
      if (div.axis() == z)
	dg = new DDDividedConsZ(div);
      break;

    case ddpolycone_rrz:
      if (div.axis() == rho)
	dg = new DDDividedPolyconeRho(div);
      else if (div.axis() == phi)
	dg = new DDDividedPolyconePhi(div);
      else if (div.axis() == z)
	dg = new DDDividedPolyconeZ(div);
      else {
	std::string s = "DDDividedAlgorithm can not divide a polycone_rrz on axis ";
	s += DDAxesNames::name(div.axis());
	s += ".\n";
	throw DDException(s);
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
	std::string s = "DDDividedAlgorithm can not divide a polyhedra_rrz on axis ";
	s += DDAxesNames::name(div.axis());
	s += ".\n";
	throw DDException(s);
      }
      break;

    case ddpolycone_rz:
    case ddpolyhedra_rz: {
      std::string s = "ERROR:  A Polycone can not be divided on any axis if it's\n";
      s += "original definition used r and z instead of ZSections. This has\n";
      s += "not (yet) been implemented.\n";
      throw DDException(s);
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
      std::string s = "DDDividedAlgorithm can not divide a ";
      s += DDSolidShapesName::name(div.parent().solid().shape());
      s += " at all (yet?).  Requested axis was ";
      s += DDAxesNames::name(div.axis());
      s += ".\n";
      throw DDException(s);
    }
      break;
    default:
      break;
    }
  return dg;

}
