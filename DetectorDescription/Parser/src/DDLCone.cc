/***************************************************************************
                          DDLCone.cc  -  description
                             -------------------
    begin                : Mon Oct 29 2001
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
#include "DetectorDescription/Parser/interface/DDLCone.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"

// DDCore dependencies
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

//#include <strstream>
#include <string>

// Default constructor.
DDLCone::DDLCone()
{
}

// Default destructor
DDLCone::~DDLCone()
{
}

// Upon encountering the end of the Cone element, call DDCore.
void DDLCone::processElement (const std::string& type, const std::string& nmspace)
{  
  DCOUT_V('P', "DDLCone::processElement started");
  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  DDXMLAttribute atts = getAttributeSet();
  try {
    DDSolid ddcone = DDSolidFactory::cons(getDDName(nmspace)
			  , ev.eval(nmspace, atts.find("dz")->second)
			  , ev.eval(nmspace, atts.find("rMin1")->second)
			  , ev.eval(nmspace, atts.find("rMax1")->second)
			  , ev.eval(nmspace, atts.find("rMin2")->second)
			  , ev.eval(nmspace, atts.find("rMax2")->second)
			  , ev.eval(nmspace, atts.find("startPhi")->second)
			  , ev.eval(nmspace, atts.find("deltaPhi")->second));
  } catch (DDException & e) {
    std::string msg = e.what();
    msg += "\nDDLCone call to DDSolidFactory failed.\n";
    throwError(msg);
  }

  DDLSolid::setReference(nmspace);

  DCOUT_V('P', "DDLCone::processElement completed");

}
