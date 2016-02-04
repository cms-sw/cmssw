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

#include "DetectorDescription/Parser/src/DDLSolid.h"

#include "DetectorDescription/Base/interface/DDdebug.h"

DDLSolid::DDLSolid( DDLElementRegistry* myreg )
  : DDXMLElement( myreg )
{}

DDLSolid::~DDLSolid( void )
{}

void
DDLSolid::setReference( const std::string& nmspace, DDCompactView& cpv )
{
  // in case it was a BooleanSolid or a ReflectionSolid, clear rSolid.
  DDXMLElement* myrSolid = myRegistry_->getElement("rSolid");
  myrSolid->clear();

  // Make sure Solid elements are in LogicalPart elements.
  if (parent() == "LogicalPart")
  {
    DDXMLElement* refsol = myRegistry_->getElement("rSolid");
    std::vector<std::string> names;
    std::vector<std::string> values;
    names.push_back("name");
    values.push_back(getAttributeSet().find("name")->second);
    refsol->loadAttributes("rSolid", names, values, nmspace, cpv);
  }

  // clear THIS solid's values.
  clear();

  DCOUT_V('P', "DDLSolid::setReference completed");
}
