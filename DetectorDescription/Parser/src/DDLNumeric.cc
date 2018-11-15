#include "DetectorDescription/Parser/src/DDLNumeric.h"
#include "DetectorDescription/Core/interface/DDConstant.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

#include <map>
#include <utility>

class DDCompactView;

DDLNumeric::DDLNumeric( DDLElementRegistry* myreg )
  : DDXMLElement( myreg )
{}
 
void
DDLNumeric::preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{}

void
DDLNumeric::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  if( parent() == "ConstantsSection" || parent() == "DDDefinition" )
  {
    DDNumeric ddnum( getDDName( nmspace ), std::make_unique<double>( myRegistry_->evaluator().eval( nmspace, getAttributeSet().find( "value" )->second )));
    clear();
  } // else, save it, don't clear it, because some other element (parent node) will use it.
}

