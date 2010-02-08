/***************************************************************************
                          DDLString.cc  -  description
                             -------------------
    begin                : Friday Nov. 21, 2003
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/




// Parser parts
#include "DDLString.h"

// other DD parts
#include "DetectorDescription/Base/interface/DDdebug.h"


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

      std::string * ts = new std::string((getAttributeSet().find("value"))->second);
      DDName ddn = getDDName(nmspace);
      DDString (ddn 
		, ts
		);

      clear();
    }

  DCOUT_V('P', "DDLString::processElement completed");
}

