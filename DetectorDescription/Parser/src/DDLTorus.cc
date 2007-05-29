
/***************************************************************************
                          DDLTorus.cc  -  description
                             -------------------
    begin                : Fri May 25 2007
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
#include "DDLTorus.h"
#include "DDLElementRegistry.h"

// DDCore dependencies
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDException.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

#include <string>

// Default constructor
DDLTorus::DDLTorus()
{
}

// Default destructor
DDLTorus::~DDLTorus()
{
}

// Upon encountering an end of the tag, call DDCore's Torus.
void DDLTorus::processElement (const std::string& name, const std::string& nmspace)
{
  DCOUT_V('P', "DDLTorus::processElement started");

  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  DDXMLAttribute atts = getAttributeSet();

  try {
        DDSolid myTorus = 
	  DDSolidFactory::torus(getDDName(nmspace)
				, ev.eval(nmspace, atts.find("innerRadius")->second)
				, ev.eval(nmspace, atts.find("outerRadius")->second)
				, ev.eval(nmspace, atts.find("torusRadius")->second)
				, ev.eval(nmspace, atts.find("startPhi")->second)
				, ev.eval(nmspace, atts.find("deltaPhi")->second)
			     );
  } catch (DDException& e) {
    std::string msg = e.what();
    msg += std::string("\nDDLParser, Call failed to DDSolidFactory.");
    throwError(msg);
  }

  DDLSolid::setReference(nmspace);

  DCOUT_V('P', "DDLTorus::processElement completed");
}
