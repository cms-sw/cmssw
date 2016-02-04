/***************************************************************************
                          DDLNumeric.cc  -  description
                             -------------------
    begin                : Friday Nov. 21, 2003
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

#include "DetectorDescription/Parser/src/DDLNumeric.h"

#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

DDLNumeric::DDLNumeric( DDLElementRegistry* myreg )
  : DDXMLElement( myreg )
{}

DDLNumeric::~DDLNumeric( void )
{}
 
void
DDLNumeric::preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{}

void
DDLNumeric::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  DCOUT_V( 'P', "DDLNumeric::processElement started" );

  if( parent() == "ConstantsSection" || parent() == "DDDefinition" )
  {
    DDNumeric ddnum( getDDName( nmspace ), new double( ExprEvalSingleton::instance().eval( nmspace, getAttributeSet().find( "value" )->second )));
    clear();
  } // else, save it, don't clear it, because some other element (parent node) will use it.

  DCOUT_V( 'P', "DDLNumeric::processElement completed" );
}

