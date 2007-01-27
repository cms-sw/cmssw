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



// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DetectorDescription/Parser/interface/DDLMaterial.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/interface/DDLLogicalPart.h"

// DDCore dependencies
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

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
void DDLMaterial::setReference (const std::string& nmspace)
{
  // in case it there were any rMaterials
  DDLElementRegistry::getElement("rMaterial")->clear();

  // Attempt to make sure Material elements can be in LogicalPart elements.
  if (DDLElementRegistry::getElement("LogicalPart")->size() > 0)
    {
      DDXMLElement* refmat = DDLElementRegistry::getElement("rMaterial");
      std::vector<std::string> names;
      std::vector<std::string> values;
      names.push_back("name");
      DDXMLAttribute atts = getAttributeSet();
      values.push_back(atts.find("name")->second);
      refmat->loadAttributes("rMaterial", names, values, nmspace);
    }
  // clear THIS material's values.
  clear();


  DCOUT_V('P', "DDLMaterial::setReference completed");
}
