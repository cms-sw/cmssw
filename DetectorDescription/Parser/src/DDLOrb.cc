/***************************************************************************
                          DDLOrb.cc  -  description
                             -------------------
    begin                : Thu Aug 19 2010
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/

#include "DetectorDescription/Parser/src/DDLOrb.h"

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

DDLOrb::DDLOrb( DDLElementRegistry* myreg )
  : DDLSolid( myreg )
{}

DDLOrb::~DDLOrb( void )
{}

// Upon encountering the end of the Orb element, call DDCore.
void
DDLOrb::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{  
  DCOUT_V( 'P', "DDLOrb::processElement started" );
  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  DDXMLAttribute atts = getAttributeSet();

  DDSolid ddorb = DDSolidFactory::orb( getDDName( nmspace ),
				       ev.eval( nmspace, atts.find( "radius" )->second ));

  DDLSolid::setReference( nmspace, cpv );

  DCOUT_V( 'P', "DDLOrb::processElement completed" );
}
