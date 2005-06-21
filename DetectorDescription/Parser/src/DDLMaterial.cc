/***************************************************************************
                          DDLMaterial.cc  -  description
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
#include "DetectorDescription/DDParser/interface/DDLMaterial.h"
#include "DetectorDescription/DDParser/interface/DDLElementRegistry.h"
#include "DetectorDescription/DDParser/interface/DDLLogicalPart.h"

// DDCore dependencies
#include "DetectorDescription/DDCore/interface/DDName.h"
#include "DetectorDescription/DDCore/interface/DDMaterial.h"
#include "DetectorDescription/DDBase/interface/DDdebug.h"

#include "DetectorDescription/DDExprAlgo/interface/ExprEvalSingleton.h"

//#include <strstream>
#include <string>

// Default constructor
DDLMaterial::DDLMaterial()
{
}

// Default desctructor
DDLMaterial::~DDLMaterial()
{
}

// Upon ending a Box element, call DDCore giving the box name, and dimensions.
void DDLMaterial::setReference (const string& nmspace)
{
  // in case it there were any rMaterials
  DDLElementRegistry::getElement("rMaterial")->clear();

  // Attempt to make sure Material elements can be in LogicalPart elements.
  if (DDLElementRegistry::getElement("LogicalPart")->size() > 0)
    {
      DDXMLElement* refmat = DDLElementRegistry::getElement("rMaterial");
      vector<string> names;
      vector<string> values;
      names.push_back("name");
      DDXMLAttribute atts = getAttributeSet();
      values.push_back(atts.find("name")->second);
      refmat->loadAttributes("rMaterial", names, values, nmspace);
    }
  // clear THIS material's values.
  clear();


  DCOUT_V('P', "DDLMaterial::setReference completed");
}
