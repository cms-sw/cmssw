/***************************************************************************
                          DDLParallelepiped.cc  -  description
                             -------------------
    begin                : Thu Aug 19 2010
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
#include "DDLParallelepiped.h"

// DDCore dependencies
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

//#include <strstream>

// Default constructor.
DDLParallelepiped::DDLParallelepiped(  DDLElementRegistry* myreg ) : DDLSolid(myreg)
{
}

// Default destructor
DDLParallelepiped::~DDLParallelepiped()
{
}

// Upon encountering the end of the Parallelepiped element, call DDCore.
void DDLParallelepiped::processElement (const std::string& name, const std::string& nmspace, DDCompactView& cpv)
{  
  DCOUT_V('P', "DDLParallelepiped::processElement started");
  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  DDXMLAttribute atts = getAttributeSet();
  DDSolid ddp = DDSolidFactory::parallelepiped(getDDName(nmspace)
					       , ev.eval(nmspace, atts.find("xHalf")->second)
					       , ev.eval(nmspace, atts.find("yHalf")->second)
					       , ev.eval(nmspace, atts.find("zHalf")->second)
					       , ev.eval(nmspace, atts.find("alpha")->second)
					       , ev.eval(nmspace, atts.find("theta")->second)
					       , ev.eval(nmspace, atts.find("phi")->second));


  DDLSolid::setReference(nmspace, cpv);

  DCOUT_V('P', "DDLParallelepiped::processElement completed");

}
