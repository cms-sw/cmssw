#include "DetectorDescription/Parser/src/DDLParallelepiped.h"

#include <map>
#include <utility>

#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLSolid.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

class DDCompactView;

DDLParallelepiped::DDLParallelepiped( DDLElementRegistry* myreg )
  : DDLSolid( myreg )
{}

// Upon encountering the end of the Parallelepiped element, call DDCore.
void
DDLParallelepiped::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{  
  ClhepEvaluator & ev = myRegistry_->evaluator();
  DDXMLAttribute atts = getAttributeSet();
  DDSolid ddp = DDSolidFactory::parallelepiped( getDDName(nmspace),
						ev.eval(nmspace, atts.find("xHalf")->second),
						ev.eval(nmspace, atts.find("yHalf")->second),
						ev.eval(nmspace, atts.find("zHalf")->second),
						ev.eval(nmspace, atts.find("alpha")->second),
						ev.eval(nmspace, atts.find("theta")->second),
						ev.eval(nmspace, atts.find("phi")->second));
  DDLSolid::setReference(nmspace, cpv);
}
