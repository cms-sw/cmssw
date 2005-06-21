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

namespace std{} using namespace std;

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DetectorDescription/DDParser/interface/DDLElementaryMaterial.h"
#include "DetectorDescription/DDParser/interface/DDLElementRegistry.h"

// DDCore dependencies
#include "DetectorDescription/DDCore/interface/DDLogicalPart.h"
#include "DetectorDescription/DDCore/interface/DDName.h"
#include "DetectorDescription/DDBase/interface/DDdebug.h"
#include "DetectorDescription/DDCore/interface/DDMaterial.h"

#include "DetectorDescription/DDExprAlgo/interface/ExprEvalSingleton.h"

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
void DDLElementaryMaterial::processElement (const string& type, const string& nmspace)
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
    string msg = e.what();
    msg += "\nDDLElementaryMaterial failed to make DDMaterial.";
    throwError(msg);
  }
  DDLMaterial::setReference(nmspace);
  clear();

  DCOUT_V('P', "DDLElementaryMaterial::processElement completed.");
}
