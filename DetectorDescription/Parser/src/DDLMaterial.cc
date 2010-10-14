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

#include "DetectorDescription/Parser/src/DDLMaterial.h"

#include "DetectorDescription/Base/interface/DDdebug.h"

DDLMaterial::DDLMaterial(  DDLElementRegistry* myreg )
  : DDXMLElement( myreg )
{}

DDLMaterial::~DDLMaterial( void )
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

  DCOUT_V('P', "DDLMaterial::setReference completed");
}
