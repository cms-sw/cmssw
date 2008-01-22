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
#include "DDLMaterial.h"
#include "DDLElementRegistry.h"

// DDCore dependencies
#include "DetectorDescription/Base/interface/DDdebug.h"


//#include <strstream>

// Default constructor
DDLMaterial::DDLMaterial()
{
}

// Default desctructor
DDLMaterial::~DDLMaterial()
{
}

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
