/***************************************************************************
                          DDLEllipsoid.cc  -  description
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
#include "DDLEllipsoid.h"

// DDCore dependencies
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

//#include <strstream>

// Default constructor.
DDLEllipsoid::DDLEllipsoid(  DDLElementRegistry* myreg ) : DDLSolid(myreg)
{
}

// Default destructor
DDLEllipsoid::~DDLEllipsoid()
{
}

// Upon encountering the end of the Ellipsoid element, call DDCore.
void DDLEllipsoid::processElement (const std::string& name, const std::string& nmspace, DDCompactView& cpv)
{  
  DCOUT_V('P', "DDLEllipsoid::processElement started");
  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  DDXMLAttribute atts = getAttributeSet();

  DDSolid ddel = DDSolidFactory::ellipsoid(getDDName(nmspace)
					   , ev.eval(nmspace, atts.find("xSemiAxis")->second)
					   , ev.eval(nmspace, atts.find("ySemiAxis")->second)
					   , ev.eval(nmspace, atts.find("zSemiAxis")->second)
					   , ev.eval(nmspace, atts.find("zBottomCut")->second)
					   , ev.eval(nmspace, atts.find("zTopCut")->second));


  DDLSolid::setReference(nmspace, cpv);

  DCOUT_V('P', "DDLEllipsoid::processElement completed");

}
