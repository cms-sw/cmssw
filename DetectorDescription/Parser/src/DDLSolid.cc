/***************************************************************************
                          DDLSolid.cc  -  description
                             -------------------
    begin                : Wed Oct 3 2002
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
#include "DetectorDescription/DDParser/interface/DDLSolid.h"
#include "DetectorDescription/DDParser/interface/DDLElementRegistry.h"
#include "DetectorDescription/DDParser/interface/DDLLogicalPart.h"

// DDCore dependencies
#include "DetectorDescription/DDCore/interface/DDName.h"
#include "DetectorDescription/DDCore/interface/DDSolid.h"
#include "DetectorDescription/DDBase/interface/DDdebug.h"

#include "DetectorDescription/DDExprAlgo/interface/ExprEvalSingleton.h"

//#include <strstream>
#include <string>

// Default constructor
DDLSolid::DDLSolid()
{
}

// Default desctructor
DDLSolid::~DDLSolid()
{
}

void DDLSolid::setReference (const string& nmspace)
{
  // in case it was a BooleanSolid or a ReflectionSolid, clear rSolid.
  DDXMLElement* myrSolid = DDLElementRegistry::getElement("rSolid");
  myrSolid->clear();

  // Make sure Solid elements are in LogicalPart elements.
  if (parent() == "LogicalPart")
    {
      DDXMLElement* refsol = DDLElementRegistry::getElement("rSolid");
      vector<string> names;
      vector<string> values;
      names.push_back("name");
      values.push_back(getAttributeSet().find("name")->second);
      refsol->loadAttributes("rSolid", names, values, nmspace);
    }

  // clear THIS solid's values.
  clear();

  DCOUT_V('P', "DDLSolid::setReference completed");
}
