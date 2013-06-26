/***************************************************************************
                          DDLTubs.cpp  -  description
                             -------------------
    begin                : Mon Oct 29 2001
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/

#include "DetectorDescription/Parser/src/DDLTubs.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDLTubs::DDLTubs( DDLElementRegistry* myreg )
  : DDLSolid( myreg )
{}

DDLTubs::~DDLTubs( void )
{}

// Upon encountering the end of a Tubs element, call DDCore.
void
DDLTubs::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  DCOUT_V('P', "DDLTubs::processElement started");

  ExprEvalInterface & ev = ExprEvalSingleton::instance();
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
					   0,
					   360*deg );
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
  else
  {
    std::string msg = "\nDDLTubs::processElement could not process element.";
    throwError(msg);
  }
  DDLSolid::setReference(nmspace, cpv);

  DCOUT_V('P', "DDLTubs::processElement completed");

}
