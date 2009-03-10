/***************************************************************************
                          DDLNumeric.cc  -  description
                             -------------------
    begin                : Friday Nov. 21, 2003
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/




// Parser parts
#include "DDLNumeric.h"
#include "DDLElementRegistry.h"

// other DD parts
#include "DetectorDescription/Core/interface/DDNumeric.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDException.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

#include <map>
#include <string>

DDLNumeric::DDLNumeric()
{
}

DDLNumeric::~DDLNumeric()
{
}
 
void DDLNumeric::preProcessElement (const std::string& name, const std::string& nmspace)
{
}

void DDLNumeric::processElement (const std::string& name, const std::string& nmspace)
{
  DCOUT_V('P', "DDLNumeric::processElement started");

  if (parent() == "ConstantsSection" || parent() == "DDDefinition")
    {
      DDNumeric ddnum ( getDDName(nmspace), new double(ExprEvalSingleton::instance().eval(nmspace, getAttributeSet().find("value")->second)) );
      clear();
    } // else, save it, don't clear it, because some other element (parent node) will use it.

  DCOUT_V('P', "DDLNumeric::processElement completed");
}

