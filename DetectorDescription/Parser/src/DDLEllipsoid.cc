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

#include "DetectorDescription/Parser/src/DDLEllipsoid.h"

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

DDLEllipsoid::DDLEllipsoid( DDLElementRegistry* myreg )
  : DDLSolid( myreg )
{}

DDLEllipsoid::~DDLEllipsoid( void )
{}

// Upon encountering the end of the Ellipsoid element, call DDCore.
void
DDLEllipsoid::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{ 
  DCOUT_V( 'P', "DDLEllipsoid::processElement started" );
  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  DDXMLAttribute atts = getAttributeSet();
  double zbot(0.), ztop(0.);
  if( atts.find( "zBottomCut" ) != atts.end() )
  {
    zbot = ev.eval( nmspace, atts.find( "zBottomCut" )->second );
  }
  if( atts.find( "zTopCut" ) != atts.end() )
  {
    ztop = ev.eval( nmspace, atts.find( "zTopCut" )->second );
  }
  DDSolid ddel = DDSolidFactory::ellipsoid( getDDName( nmspace ),
					    ev.eval(nmspace, atts.find("xSemiAxis")->second),
					    ev.eval(nmspace, atts.find("ySemiAxis")->second),
					    ev.eval(nmspace, atts.find("zSemiAxis")->second),
					    zbot,
					    ztop );
  DDLSolid::setReference( nmspace, cpv );

  DCOUT_V( 'P', "DDLEllipsoid::processElement completed" );
}
