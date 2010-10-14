/***************************************************************************
                          DDLPseudoTrap.cc  -  description
                             -------------------
    begin                : Mon Jul 17 2003
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/

#include "DetectorDescription/Parser/src/DDLPseudoTrap.h"

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

DDLPseudoTrap::DDLPseudoTrap( DDLElementRegistry* myreg )
  : DDLSolid( myreg )
{}

DDLPseudoTrap::~DDLPseudoTrap( void )
{}

// Upon encountering an end of the tag, call DDCore's Trap.
void
DDLPseudoTrap::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  DCOUT_V('P', "DDLPseudoTrap::processElement started");

  ExprEvalInterface & ev = ExprEvalSingleton::instance();
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

  DCOUT_V('P', "DDLPseudoTrap::processElement completed");
}
