/***************************************************************************
                          DDLString.cc  -  description
                             -------------------
    begin                : Friday Nov. 21, 2003
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/




// Parser parts
#include "DetectorDescription/Parser/interface/DDLString.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"

// other DD parts
#include "DetectorDescription/Core/interface/DDString.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDException.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

#include <map>
#include <string>

DDLString::DDLString()
{
}

DDLString::~DDLString()
{
}
 
void DDLString::preProcessElement (const std::string& name, const std::string& nmspace)
{
}

void DDLString::processElement (const std::string& name, const std::string& nmspace)
{
  DCOUT_V('P', "DDLString::processElement started");
  if (parent() == "ConstantsSection" || parent() == "DDDefinition")
    {
      // I do not like "newing" things without my control.  But this is
      // the only way I was able to get this to work.
      try {
	std::string * ts = new std::string((getAttributeSet().find("value"))->second);
	DDName ddn = getDDName(nmspace);
	DDString (ddn 
		  , ts
		  );
      } catch (DDException & e) {
	std::string msg(e.what());
        msg += "\nDDLString failed to create a DDString.";
	throwError(msg);
      }

      clear();
    }

  DCOUT_V('P', "DDLString::processElement completed");
}

