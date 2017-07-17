#include "DetectorDescription/Parser/src/DDLOrb.h"

#include <map>
#include <utility>

#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLSolid.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

class DDCompactView;

DDLOrb::DDLOrb( DDLElementRegistry* myreg )
  : DDLSolid( myreg )
{}

// Upon encountering the end of the Orb element, call DDCore.
void
DDLOrb::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{  
  ClhepEvaluator & ev = myRegistry_->evaluator();
  DDXMLAttribute atts = getAttributeSet();

  DDSolid ddorb = DDSolidFactory::orb( getDDName( nmspace ),
				       ev.eval( nmspace, atts.find( "radius" )->second ));

  DDLSolid::setReference( nmspace, cpv );
}
