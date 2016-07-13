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

#include <map>
#include <utility>

#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/ExprAlgo/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLSolid.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

class DDCompactView;

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
  ClhepEvaluator & ev = myRegistry_->evaluator();
  DDXMLAttribute atts = getAttributeSet();

  DDSolid ddorb = DDSolidFactory::orb( getDDName( nmspace ),
				       ev.eval( nmspace, atts.find( "radius" )->second ));

  DDLSolid::setReference( nmspace, cpv );

  DCOUT_V( 'P', "DDLOrb::processElement completed" );
}
