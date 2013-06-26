
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

#include "DetectorDescription/Parser/src/DDLTorus.h"

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

DDLTorus::DDLTorus( DDLElementRegistry* myreg )
  : DDLSolid( myreg )
{}

DDLTorus::~DDLTorus( void )
{}

// Upon encountering an end of the tag, call DDCore's Torus.
void
DDLTorus::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  DCOUT_V('P', "DDLTorus::processElement started");

  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  DDXMLAttribute atts = getAttributeSet();

  DDSolid myTorus = 
    DDSolidFactory::torus( getDDName(nmspace),
			   ev.eval(nmspace, atts.find("innerRadius")->second),
			   ev.eval(nmspace, atts.find("outerRadius")->second),
			   ev.eval(nmspace, atts.find("torusRadius")->second),
			   ev.eval(nmspace, atts.find("startPhi")->second),
			   ev.eval(nmspace, atts.find("deltaPhi")->second));
  DDLSolid::setReference( nmspace, cpv );

  DCOUT_V('P', "DDLTorus::processElement completed");
}
