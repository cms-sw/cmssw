#include "DetectorDescription/Parser/src/DDLMaterial.h"

#include <map>
#include <utility>
#include <vector>

#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

class DDCompactView;

DDLMaterial::DDLMaterial(  DDLElementRegistry* myreg )
  : DDXMLElement( myreg )
{}

void
DDLMaterial::setReference( const std::string& nmspace, DDCompactView& cpv )
{
  // in case it there were any rMaterials
  myRegistry_->getElement("rMaterial")->clear();

  // Attempt to make sure Material elements can be in LogicalPart elements.
  if (myRegistry_->getElement("LogicalPart")->size() > 0)
    {
      DDXMLElement* refmat = myRegistry_->getElement("rMaterial");
      std::vector<std::string> names;
      std::vector<std::string> values;
      names.push_back("name");
      DDXMLAttribute atts = getAttributeSet();
      values.push_back(atts.find("name")->second);
      refmat->loadAttributes("rMaterial", names, values, nmspace, cpv);
    }
  // clear THIS material's values.
  clear();
}
