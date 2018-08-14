#include "DetectorDescription/Parser/src/DDLTubs.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Core/interface/DDUnits.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLSolid.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

#include <map>
#include <utility>

class DDCompactView;

using namespace dd::operators;

DDLTubs::DDLTubs( DDLElementRegistry* myreg )
  : DDLSolid( myreg )
{}

// Upon encountering the end of a Tubs element, call DDCore.
void
DDLTubs::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  ClhepEvaluator & ev = myRegistry_->evaluator();
  DDXMLAttribute atts = getAttributeSet();
  bool cutInside(false);

  if (atts.find("cutInside") != atts.end()) {
    cutInside = (atts.find("cutInside")->second == "true") ? true : false;
  }

  if (name == "Tubs")
  {
    DDSolid myTubs = DDSolidFactory::tubs( getDDName(nmspace),
					   ev.eval(nmspace, atts.find("dz")->second),
					   ev.eval(nmspace, atts.find("rMin")->second),
					   ev.eval(nmspace, atts.find("rMax")->second),
					   ev.eval(nmspace, atts.find("startPhi")->second),
					   ev.eval(nmspace, atts.find("deltaPhi")->second));
  }
  else if (name == "Tube")
  {
    DDSolid myTubs = DDSolidFactory::tubs( getDDName(nmspace),
					   ev.eval(nmspace, atts.find("dz")->second),
					   ev.eval(nmspace, atts.find("rMin")->second),
					   ev.eval(nmspace, atts.find("rMax")->second),
					   0_deg,
					   360_deg );
  }
  else if (name == "TruncTubs")
  {      
    DDSolid myTT = DDSolidFactory::truncTubs( getDDName(nmspace),
					      ev.eval(nmspace, atts.find("zHalf")->second),
					      ev.eval(nmspace, atts.find("rMin")->second),
					      ev.eval(nmspace, atts.find("rMax")->second),
					      ev.eval(nmspace, atts.find("startPhi")->second), //0. // startPhi
					      ev.eval(nmspace, atts.find("deltaPhi")->second),
					      ev.eval(nmspace, atts.find("cutAtStart")->second),
					      ev.eval(nmspace, atts.find("cutAtDelta")->second),
					      cutInside ); // cutInside
  }
  else if (name == "CutTubs")
  {
    DDSolid myTubs = DDSolidFactory::cuttubs( getDDName(nmspace),
					      ev.eval(nmspace, atts.find("dz")->second),
					      ev.eval(nmspace, atts.find("rMin")->second),
					      ev.eval(nmspace, atts.find("rMax")->second),
					      ev.eval(nmspace, atts.find("startPhi")->second),
					      ev.eval(nmspace, atts.find("deltaPhi")->second),
					      ev.eval(nmspace, atts.find("lx")->second),
					      ev.eval(nmspace, atts.find("ly")->second),
					      ev.eval(nmspace, atts.find("lz")->second),
					      ev.eval(nmspace, atts.find("tx")->second),
					      ev.eval(nmspace, atts.find("ty")->second),
					      ev.eval(nmspace, atts.find("tz")->second));
  }
  else
  {
    std::string msg = "\nDDLTubs::processElement could not process element.";
    throwError(msg);
  }
  DDLSolid::setReference(nmspace, cpv);
}
