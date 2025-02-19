/***************************************************************************
                          DDLCone.cc  -  description
                             -------------------
    begin                : Mon Oct 29 2001
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/

#include "DetectorDescription/Parser/src/DDLCone.h"

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

DDLCone::DDLCone( DDLElementRegistry* myreg )
  : DDLSolid( myreg )
{}

DDLCone::~DDLCone( void )
{}

// Upon encountering the end of the Cone element, call DDCore.
void
DDLCone::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{  
  DCOUT_V( 'P', "DDLCone::processElement started" );
  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  DDXMLAttribute atts = getAttributeSet();

  DDSolid ddcone = DDSolidFactory::cons( getDDName( nmspace ),
					 ev.eval( nmspace, atts.find( "dz" )->second ),
					 ev.eval( nmspace, atts.find( "rMin1" )->second ),
					 ev.eval( nmspace, atts.find( "rMax1" )->second ),
					 ev.eval( nmspace, atts.find( "rMin2" )->second ),
					 ev.eval( nmspace, atts.find( "rMax2" )->second ),
					 ev.eval( nmspace, atts.find( "startPhi" )->second ),
					 ev.eval( nmspace, atts.find( "deltaPhi" )->second ));

  DDLSolid::setReference( nmspace, cpv );

  DCOUT_V( 'P', "DDLCone::processElement completed" );
}
