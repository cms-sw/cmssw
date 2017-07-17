#include "DetectorDescription/Parser/src/DDLEllipticalTube.h"

#include <map>
#include <utility>

#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLSolid.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

class DDCompactView;

DDLEllipticalTube::DDLEllipticalTube( DDLElementRegistry* myreg )
  : DDLSolid( myreg )
{}

// Upon encountering the end of the EllipticalTube element, call DDCore.
void
DDLEllipticalTube::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{  
  ClhepEvaluator & ev = myRegistry_->evaluator();
  DDXMLAttribute atts = getAttributeSet();

  DDSolid ddet = DDSolidFactory::ellipticalTube( getDDName( nmspace ),
						 ev.eval( nmspace, atts.find( "xSemiAxis" )->second ),
						 ev.eval( nmspace, atts.find( "ySemiAxis" )->second ),
						 ev.eval( nmspace, atts.find( "zHeight" )->second ));
  DDLSolid::setReference( nmspace, cpv );
}
