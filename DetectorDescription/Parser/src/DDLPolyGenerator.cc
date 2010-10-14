/***************************************************************************
                          DDLPolyGenerator.cpp  -  description
                             -------------------
    begin                : Thu Oct 25 2001
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/

#include "DetectorDescription/Parser/src/DDLPolyGenerator.h"

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

DDLPolyGenerator::DDLPolyGenerator( DDLElementRegistry* myreg )
  : DDLSolid( myreg )
{}

DDLPolyGenerator::~DDLPolyGenerator( void )
{}

void
DDLPolyGenerator::preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  myRegistry_->getElement("RZPoint")->clear();
  myRegistry_->getElement("ZSection")->clear();
}

// Upon encountering an end Polycone or Polyhedra tag, process the RZPoints
// element and extract the r and z std::vectors to feed into DDCore.  Then, clear
// the RZPoints and clear this element.
void
DDLPolyGenerator::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  DCOUT_V('P', "DDLPolyGenerator::processElement started");

  DDXMLElement* myRZPoints = myRegistry_->getElement("RZPoint");
  DDXMLElement* myZSection = myRegistry_->getElement("ZSection");

  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  DDXMLAttribute atts;

  // get z and r
  std::vector<double> z, r;
  for (size_t i = 0; i < myRZPoints->size(); ++i)
  {
    atts = myRZPoints->getAttributeSet(i);
    z.push_back(ev.eval(nmspace, atts.find("z")->second));
    r.push_back(ev.eval(nmspace, atts.find("r")->second));
  }

  // if z is empty, then it better not have been a polycone defined
  // by RZPoints, instead, it must be a ZSection defined polycone.
  if (z.size() == 0 )
  {
    // get zSection information, note, we already have a z declared above
    // and we will use r for rmin.  In this case, no use "trying" because
    // it better be there!
    std::vector<double> rMax;

    for (size_t i = 0; i < myZSection->size(); ++i)
    {
      atts = myZSection->getAttributeSet(i);
      z.push_back(ev.eval(nmspace, atts.find("z")->second));
      r.push_back(ev.eval(nmspace, atts.find("rMin")->second));
      rMax.push_back(ev.eval(nmspace, atts.find("rMax")->second));
    }
    atts = getAttributeSet();
    if (name == "Polycone") // defined with ZSections 
    {
      DDSolid ddpolycone = 
	DDSolidFactory::polycone(getDDName(nmspace)
				 , ev.eval(nmspace, atts.find("startPhi")->second)
				 , ev.eval(nmspace, atts.find("deltaPhi")->second)
				 , z
				 , r
				 , rMax);
    }
    else if (name == "Polyhedra")  // defined with ZSections
    {
      DDSolid ddpolyhedra = 
	DDSolidFactory::polyhedra(getDDName(nmspace)
				  , int (ev.eval(nmspace, atts.find("numSide")->second))
				  , ev.eval(nmspace, atts.find("startPhi")->second)
				  , ev.eval(nmspace, atts.find("deltaPhi")->second)
				  , z
				  , r
				  , rMax);
    }

  }
  else if (name == "Polycone") // defined with RZPoints
  {
    atts = getAttributeSet();
    DDSolid ddpolycone = 
      DDSolidFactory::polycone(getDDName(nmspace)
			       , ev.eval(nmspace, atts.find("startPhi")->second)
			       , ev.eval(nmspace, atts.find("deltaPhi")->second)
			       , z
			       , r);
  }
  else if (name == "Polyhedra") // defined with RZPoints
  {
    atts = getAttributeSet();
    DDSolid ddpolyhedra = 
      DDSolidFactory::polyhedra(getDDName(nmspace)
				, int (ev.eval(nmspace, atts.find("numSide")->second))
				, ev.eval(nmspace, atts.find("startPhi")->second)
				, ev.eval(nmspace, atts.find("deltaPhi")->second)
				, z
				, r);
  }
  else
  {
    std::string msg = "\nDDLPolyGenerator::processElement was called with incorrect name of solid: " + name;
    throwError(msg);
  }
  DDLSolid::setReference(nmspace, cpv);

  // clear out RZPoint element accumulator and ZSections
  myRZPoints->clear();
  myZSection->clear();
  clear();

  DCOUT_V('P', "DDLPolyGenerator::processElement completed");
}
