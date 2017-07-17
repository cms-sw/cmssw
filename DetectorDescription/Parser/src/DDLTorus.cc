#include "DetectorDescription/Parser/src/DDLTorus.h"

#include <map>
#include <utility>

#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLSolid.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

class DDCompactView;

DDLTorus::DDLTorus( DDLElementRegistry* myreg )
  : DDLSolid( myreg )
{}

// Upon encountering an end of the tag, call DDCore's Torus.
void
DDLTorus::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  ClhepEvaluator & ev = myRegistry_->evaluator();
  DDXMLAttribute atts = getAttributeSet();

  DDSolid myTorus = 
    DDSolidFactory::torus( getDDName(nmspace),
			   ev.eval(nmspace, atts.find("innerRadius")->second),
			   ev.eval(nmspace, atts.find("outerRadius")->second),
			   ev.eval(nmspace, atts.find("torusRadius")->second),
			   ev.eval(nmspace, atts.find("startPhi")->second),
			   ev.eval(nmspace, atts.find("deltaPhi")->second));
  DDLSolid::setReference( nmspace, cpv );
}
