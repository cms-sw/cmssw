/***************************************************************************
                          DDLEllipticalTube.cc  -  description
                             -------------------
    begin                : Thu Aug 19 2010
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/

#include "DetectorDescription/Parser/src/DDLEllipticalTube.h"

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

DDLEllipticalTube::DDLEllipticalTube( DDLElementRegistry* myreg )
  : DDLSolid( myreg )
{}

DDLEllipticalTube::~DDLEllipticalTube( void )
{}

// Upon encountering the end of the EllipticalTube element, call DDCore.
void
DDLEllipticalTube::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{  
  DCOUT_V( 'P', "DDLEllipticalTube::processElement started" );
  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  DDXMLAttribute atts = getAttributeSet();

  DDSolid ddet = DDSolidFactory::ellipticalTube( getDDName( nmspace ),
						 ev.eval( nmspace, atts.find( "xSemiAxis" )->second ),
						 ev.eval( nmspace, atts.find( "ySemiAxis" )->second ),
						 ev.eval( nmspace, atts.find( "zHeight" )->second ));
  DDLSolid::setReference( nmspace, cpv );

  DCOUT_V( 'P', "DDLEllipticalTube::processElement completed" );
}
