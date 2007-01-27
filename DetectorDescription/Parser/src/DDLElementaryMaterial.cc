/***************************************************************************
                          DDLElementaryMaterial.cc  -  description
                             -------------------
    begin                : Wed Oct 31 2001
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/



// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DetectorDescription/Parser/interface/DDLElementaryMaterial.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"

// DDCore dependencies
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

#include <string>
#include <iostream>

// Default constructor.
DDLElementaryMaterial::DDLElementaryMaterial()
{
}

// Default destructor.
DDLElementaryMaterial::~DDLElementaryMaterial()
{
}

// Upon encountering an end of an ElementaryMaterial element, we call DDCore
void DDLElementaryMaterial::processElement (const std::string& type, const std::string& nmspace)
{
  DCOUT_V('P', "DDLElementaryMaterial::processElement started");

  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  DDXMLAttribute atts = getAttributeSet();
  try {
    DDMaterial mat = DDMaterial(getDDName(nmspace)
      , ev.eval(nmspace, atts.find("atomicNumber")->second)
      , ev.eval(nmspace, atts.find("atomicWeight")->second)
      , ev.eval(nmspace, atts.find("density")->second));
  } catch (DDException & e) {
    std::string msg = e.what();
    msg += "\nDDLElementaryMaterial failed to make DDMaterial.";
    throwError(msg);
  }
  DDLMaterial::setReference(nmspace);
  clear();

  DCOUT_V('P', "DDLElementaryMaterial::processElement completed.");
}
