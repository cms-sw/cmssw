/***************************************************************************
                          DDLNumeric.cc  -  description
                             -------------------
    begin                : Friday Nov. 21, 2003
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/


namespace std{} using namespace std;

// Parser parts
#include "DetectorDescription/DDParser/interface/DDLNumeric.h"
#include "DetectorDescription/DDParser/interface/DDLElementRegistry.h"

// other DD parts
#include "DetectorDescription/DDCore/interface/DDNumeric.h"
#include "DetectorDescription/DDBase/interface/DDdebug.h"
#include "DetectorDescription/DDBase/interface/DDException.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include "DetectorDescription/DDExprAlgo/interface/ExprEvalSingleton.h"

#include <map>
#include <string>

DDLNumeric::DDLNumeric()
{
}

DDLNumeric::~DDLNumeric()
{
}
 
void DDLNumeric::preProcessElement (const string& name, const string& nmspace)
{
}

void DDLNumeric::processElement (const string& name, const string& nmspace)
{
  DCOUT_V('P', "DDLNumeric::processElement started");

  if (parent() == "ConstantsSection" || parent() == "DDDefinition")
    {
      try {
	DDNumeric ddnum ( getDDName(nmspace), new double(ExprEvalSingleton::instance().eval(nmspace, getAttributeSet().find("value")->second)) );
      } catch (DDException & e) {
	string msg(e.what());
	msg += "\nDDLNumeric failed to create a DDNumeric.";
	throwError(msg);
      }  
      clear();
    }

  DCOUT_V('P', "DDLNumeric::processElement completed");
}

