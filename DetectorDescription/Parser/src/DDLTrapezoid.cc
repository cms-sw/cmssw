#include "DetectorDescription/Parser/src/DDLTrapezoid.h"

#include <map>
#include <utility>

#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLSolid.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

class DDCompactView;

DDLTrapezoid::DDLTrapezoid( DDLElementRegistry* myreg )
  : DDLSolid( myreg )
{}

// Upon encountering an end of the tag, call DDCore's Trap.
void
DDLTrapezoid::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  ClhepEvaluator & ev = myRegistry_->evaluator();

  DDXMLAttribute atts = getAttributeSet();

  double phi = 0.0;
  double theta = 0.0;
  double dy2 = 0.0;

  if (atts.find("phi") != atts.end())
    phi = ev.eval(nmspace, atts.find("phi")->second);

  if (atts.find("theta") != atts.end())
    theta = ev.eval(nmspace, atts.find("theta")->second);

  if (atts.find("dy2") != atts.end())
    dy2 = ev.eval(nmspace, atts.find("dy2")->second);
  else if (atts.find("dy1") != atts.end())
    dy2 = ev.eval(nmspace, atts.find("dy1")->second);

  if (name == "Trapezoid")
  {
    DDSolid myTrap = 
      DDSolidFactory::trap( getDDName(nmspace),
			    ev.eval(nmspace, atts.find("dz")->second),
			    theta,
			    phi,
			    ev.eval(nmspace, atts.find("h1")->second),
			    ev.eval(nmspace, atts.find("bl1")->second),
			    ev.eval(nmspace, atts.find("tl1")->second),
			    ev.eval(nmspace, atts.find("alp1")->second),
			    ev.eval(nmspace, atts.find("h2")->second),
			    ev.eval(nmspace, atts.find("bl2")->second),
			    ev.eval(nmspace, atts.find("tl2")->second),
			    ev.eval(nmspace, atts.find("alp2")->second));
  }
  else if (name == "Trd1") 
  {
    DDSolid myTrd1 = 
      DDSolidFactory::trap( getDDName(nmspace),
			    ev.eval(nmspace, atts.find("dz")->second),
			    0,
			    0,
			    ev.eval(nmspace, atts.find("dy1")->second),
			    ev.eval(nmspace, atts.find("dx1")->second),
			    ev.eval(nmspace, atts.find("dx1")->second),
			    0,
			    dy2,
			    ev.eval(nmspace, atts.find("dx2")->second),
			    ev.eval(nmspace, atts.find("dx2")->second),
			    0 );
  }
  else
  {
    std::string msg = "\nDDLTrapezoid::processElement failed to process element of name: " 
		      + name
		      + ".  It can only process Trapezoid and Trd1.";
    throwError(msg);
  }

  DDLSolid::setReference( nmspace, cpv );
}
