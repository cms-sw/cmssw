/***************************************************************************
                          DDLPseudoTrap.cc  -  description
                             -------------------
    begin                : Mon Jul 17 2003
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/



// -------------------------------------------------------------------------
// Includes 
// -------------------------------------------------------------------------
// Parser parts
#include "DetectorDescription/Parser/interface/DDLPseudoTrap.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"

// DDCore dependencies
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDException.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

#include <string>

// Default constructor
DDLPseudoTrap::DDLPseudoTrap()
{
}

// Default destructor
DDLPseudoTrap::~DDLPseudoTrap()
{
}

// Upon encountering an end of the tag, call DDCore's Trap.
void DDLPseudoTrap::processElement (const std::string& type, const std::string& nmspace)
{
  DCOUT_V('P', "DDLPseudoTrap::processElement started");

  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  DDXMLAttribute atts = getAttributeSet();

  try {
    DDSolid myTrap = DDSolidFactory::pseudoTrap(getDDName(nmspace)
		      , ev.eval(nmspace, atts.find("dx1")->second)
		      , ev.eval(nmspace, atts.find("dx2")->second)
		      , ev.eval(nmspace, atts.find("dy1")->second)
		      , ev.eval(nmspace, atts.find("dy2")->second)
		      , ev.eval(nmspace, atts.find("dz")->second)
		      , ev.eval(nmspace, atts.find("radius")->second)
		      , (atts.find("atMinusZ")->second == "true") ? true : false
		      );
  } catch (DDException & e) {
    std::string msg(e.what());
    msg += "\nDDLPseudoTrap failed call to DDSolidFactory.";
    throwError(msg);
  }

  DDLSolid::setReference(nmspace);

  DCOUT_V('P', "DDLPseudoTrap::processElement completed");
}
