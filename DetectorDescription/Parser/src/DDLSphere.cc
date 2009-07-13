/***************************************************************************
                          DDLSphere.cc  -  description
                             -------------------
    begin                : Sun July 12 2009
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
#include "DDLSphere.h"

// DDCore dependencies
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

//#include <strstream>

// Default constructor.
DDLSphere::DDLSphere()
{
}

// Default destructor
DDLSphere::~DDLSphere()
{
}

// Upon encountering the end of the Sphere element, call DDCore.
void DDLSphere::processElement (const std::string& type, const std::string& nmspace)
{  
  DCOUT_V('P', "DDLSphere::processElement started");
  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  DDXMLAttribute atts = getAttributeSet();

  DDSolid ddspere = DDSolidFactory::sphere(getDDName(nmspace)
					   , ev.eval(nmspace, atts.find("innerRadius")->second)
					   , ev.eval(nmspace, atts.find("outerRadius")->second)
					   , ev.eval(nmspace, atts.find("startPhi")->second)
					   , ev.eval(nmspace, atts.find("deltaPhi")->second)
					   , ev.eval(nmspace, atts.find("startTheta")->second)
					   , ev.eval(nmspace, atts.find("deltaTheta")->second));


  DDLSolid::setReference(nmspace);

  DCOUT_V('P', "DDLSphere::processElement completed");

}
