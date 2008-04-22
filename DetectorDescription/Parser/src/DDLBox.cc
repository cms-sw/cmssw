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
#include "DDLBox.h"
#include "DDLElementRegistry.h"
#include "DDLLogicalPart.h"

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
  
  DDName ddname = getDDName(nmspace);
  DDSolid ddbox = DDSolidFactory::box(ddname
				      , ev.eval(nmspace, atts.find("dx")->second)
				      , ev.eval(nmspace, atts.find("dy")->second)
				      , ev.eval(nmspace, atts.find("dz")->second));
  // Attempt to make sure Solid elements can be in LogicalPart elements.
  DDLSolid::setReference(nmspace);

  DCOUT_V('P', "DDLBox::processElement completed");
}
