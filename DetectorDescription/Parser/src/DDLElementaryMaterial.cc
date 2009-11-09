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
#include "DDLElementaryMaterial.h"

// DDCore dependencies
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

#include <iostream>

// Default constructor.
DDLElementaryMaterial::DDLElementaryMaterial(  DDLElementRegistry* myreg ) : DDLMaterial(myreg)
{
}

// Default destructor.
DDLElementaryMaterial::~DDLElementaryMaterial()
{
}

// Upon encountering an end of an ElementaryMaterial element, we call DDCore
void DDLElementaryMaterial::processElement (const std::string& name, const std::string& nmspace, DDCompactView& cpv)
{
  DCOUT_V('P', "DDLElementaryMaterial::processElement started");

  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  DDXMLAttribute atts = getAttributeSet();

  DDMaterial mat = DDMaterial(getDDName(nmspace)
			      , ev.eval(nmspace, atts.find("atomicNumber")->second)
			      , ev.eval(nmspace, atts.find("atomicWeight")->second)
			      , ev.eval(nmspace, atts.find("density")->second));

  DDLMaterial::setReference(nmspace, cpv);
  clear();

  DCOUT_V('P', "DDLElementaryMaterial::processElement completed.");
}
