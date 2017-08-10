#include "DetectorDescription/Parser/src/DDLSphere.h"

#include <map>
#include <utility>

#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLSolid.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

class DDCompactView;

DDLSphere::DDLSphere( DDLElementRegistry* myreg )
  : DDLSolid( myreg )
{}

// Upon encountering the end of the Sphere element, call DDCore.
void
DDLSphere::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{  
  ClhepEvaluator & ev = myRegistry_->evaluator();
  DDXMLAttribute atts = getAttributeSet();
  DDSolid ddsphere = DDSolidFactory::sphere( getDDName(nmspace),
					     ev.eval(nmspace, atts.find("innerRadius")->second),
					     ev.eval(nmspace, atts.find("outerRadius")->second),
					     ev.eval(nmspace, atts.find("startPhi")->second),
					     ev.eval(nmspace, atts.find("deltaPhi")->second),
					     ev.eval(nmspace, atts.find("startTheta")->second),
					     ev.eval(nmspace, atts.find("deltaTheta")->second ));
  
  DDLSolid::setReference(nmspace, cpv);
}
