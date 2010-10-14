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

#include "DetectorDescription/Parser/src/DDLSphere.h"

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

DDLSphere::DDLSphere( DDLElementRegistry* myreg )
  : DDLSolid( myreg )
{}

DDLSphere::~DDLSphere( void )
{}

// Upon encountering the end of the Sphere element, call DDCore.
void
DDLSphere::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{  
  DCOUT_V('P', "DDLSphere::processElement started");
  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  DDXMLAttribute atts = getAttributeSet();
  DDSolid ddsphere = DDSolidFactory::sphere( getDDName(nmspace),
					     ev.eval(nmspace, atts.find("innerRadius")->second),
					     ev.eval(nmspace, atts.find("outerRadius")->second),
					     ev.eval(nmspace, atts.find("startPhi")->second),
					     ev.eval(nmspace, atts.find("deltaPhi")->second),
					     ev.eval(nmspace, atts.find("startTheta")->second),
					     ev.eval(nmspace, atts.find("deltaTheta")->second ));
  
  DDLSolid::setReference(nmspace, cpv);

  DCOUT_V('P', "DDLSphere::processElement completed");
}
