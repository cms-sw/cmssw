#include "DetectorDescription/Parser/src/DDLPseudoTrap.h"

#include <map>
#include <utility>

#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLSolid.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

class DDCompactView;

DDLPseudoTrap::DDLPseudoTrap( DDLElementRegistry* myreg )
  : DDLSolid( myreg )
{}

// Upon encountering an end of the tag, call DDCore's Trap.
void
DDLPseudoTrap::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  ClhepEvaluator & ev = myRegistry_->evaluator();
  DDXMLAttribute atts = getAttributeSet();

  DDSolid myTrap = DDSolidFactory::pseudoTrap( getDDName(nmspace),
					       ev.eval(nmspace, atts.find("dx1")->second),
					       ev.eval(nmspace, atts.find("dx2")->second),
					       ev.eval(nmspace, atts.find("dy1")->second),
					       ev.eval(nmspace, atts.find("dy2")->second),
					       ev.eval(nmspace, atts.find("dz")->second),
					       ev.eval(nmspace, atts.find("radius")->second),
					       (atts.find("atMinusZ")->second == "true") ? true : false );

  DDLSolid::setReference(nmspace, cpv);
}
