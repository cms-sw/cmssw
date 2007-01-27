/***************************************************************************
                          DDLNumeric.cc  -  description
                             -------------------
    begin                : Friday Nov. 21, 2003
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/




// Parser parts
#include "DetectorDescription/Parser/interface/DDLNumeric.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"

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
      try {
	DDNumeric ddnum ( getDDName(nmspace), new double(ExprEvalSingleton::instance().eval(nmspace, getAttributeSet().find("value")->second)) );
      } catch (DDException & e) {
	std::string msg(e.what());
	msg += "\nDDLNumeric failed to create a DDNumeric.";
	throwError(msg);
      }  
      clear();
    }

  DCOUT_V('P', "DDLNumeric::processElement completed");
}

