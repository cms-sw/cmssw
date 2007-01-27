/***************************************************************************
                          DDLBox.cc  -  description
                             -------------------
    begin                : Wed Oct 24 2001
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
#include "DetectorDescription/Parser/interface/DDLBox.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/interface/DDLLogicalPart.h"

// DDCore dependencies
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

//#include <strstream>
#include <string>

// Default constructor
DDLBox::DDLBox()
{
}

// Default desctructor
DDLBox::~DDLBox()
{
}

// Upon ending a Box element, call DDCore giving the box name, and dimensions.
void DDLBox::processElement (const std::string& type, const std::string& nmspace)
{
  DCOUT_V('P', "DDLBox::processElement started");
  
  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  DDXMLAttribute atts = getAttributeSet();
  
  try {
    DDName ddname = getDDName(nmspace);
    DDSolid ddbox = DDSolidFactory::box(ddname
			, ev.eval(nmspace, atts.find("dx")->second)
			, ev.eval(nmspace, atts.find("dy")->second)
			, ev.eval(nmspace, atts.find("dz")->second));
    // Attempt to make sure Solid elements can be in LogicalPart elements.
    DDLSolid::setReference(nmspace);
  } catch (DDException& e)
    {
      std::string msg(e.what());
      msg += std::string("\n\tDDLParser, Call failed to DDSolidFactory.");
      throwError(msg);
    }

  DCOUT_V('P', "DDLBox::processElement completed");
}
