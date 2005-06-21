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

namespace std{} using namespace std;

// -------------------------------------------------------------------------
// Includes 
// -------------------------------------------------------------------------
// Parser parts
#include "DetectorDescription/DDParser/interface/DDLPseudoTrap.h"
#include "DetectorDescription/DDParser/interface/DDLElementRegistry.h"

// DDCore dependencies
#include "DetectorDescription/DDCore/interface/DDName.h"
#include "DetectorDescription/DDCore/interface/DDSolid.h"
#include "DetectorDescription/DDBase/interface/DDdebug.h"
#include "DetectorDescription/DDBase/interface/DDException.h"

#include "DetectorDescription/DDExprAlgo/interface/ExprEvalSingleton.h"

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
void DDLPseudoTrap::processElement (const string& type, const string& nmspace)
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
    string msg(e.what());
    msg += "\nDDLPseudoTrap failed call to DDSolidFactory.";
    throwError(msg);
  }

  DDLSolid::setReference(nmspace);

  DCOUT_V('P', "DDLPseudoTrap::processElement completed");
}
