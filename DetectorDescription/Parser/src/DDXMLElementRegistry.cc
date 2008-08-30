/***************************************************************************
                          DDXMLElementRegistry.cc  -  description
                             -------------------
    begin                : Wed Mar 27 2002
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
#include "DDXMLElementRegistry.h"

// DDCore dependencies
#include "DetectorDescription/Base/interface/DDdebug.h"

#include <string>
#include <algorithm>
#include <map>
#include <vector>
#include <iostream>

// -------------------------------------------------------------------------
// Constructor/Destructor
// -------------------------------------------------------------------------

DDXMLElementRegistry::DDXMLElementRegistry()
{
  //  registry_ = new RegistryMap;
}

DDXMLElementRegistry::~DDXMLElementRegistry()
{
  // Complicated cleanup.  I keep track of DDXMLElements that have
  // already been deleted using this vector.  Then delete them one-by-one.
  std::vector<DDXMLElement*> toDelete;
  for ( RegistryMap::const_iterator it = registry_.begin(); it != registry_.end(); ++it) {
    std::vector<DDXMLElement*>::const_iterator deleteIt = std::find(toDelete.begin(), toDelete.end(), it->second);
    if ( deleteIt == toDelete.end() ) {
      toDelete.push_back(it->second);
      delete it->second;
    }
  }
}

// -------------------------------------------------------------------------
// Implementation
// -------------------------------------------------------------------------

//  This allows Elements to register themselves with the static registry
void DDXMLElementRegistry::registerElement(const std::string& name, DDXMLElement* element)
{
  DCOUT_V('P',"DDXMLElementRegistry::registerElementBase: "  << name << " at " << element);
  registry_[name] = element;
}

// THE most important part.  Getting the pointer to a given element type.
DDXMLElement* DDXMLElementRegistry::getElement(const std::string& name)
{
  DCOUT_V('P',  "DDXMLElementRegistry::getElement " << name << std::endl);
  RegistryMap::iterator it = registry_.find(name);
  DDXMLElement* myret = NULL;
  if (it != registry_.end())
    myret = it->second;
  DCOUT_V('P',  "DDXMLElementRegistry::getElement " << name << std::endl);
  return myret;
}

// Get the name given a pointer.  This may not be needed...
std::string DDXMLElementRegistry::getElementName(DDXMLElement* theElementBase)
{
  std::string ret = "";
  for (RegistryMap::const_iterator it = registry_.begin(); it != registry_.end(); ++it)
    if (it->second == theElementBase)
      ret = it->first;
  return ret;
}

std::ostream & operator<<(std::ostream & os, const DDXMLElementRegistry & element)
{
  element.stream(os);
  return os;
}

void DDXMLElementRegistry::stream(std::ostream & os) const
{
  os << "Output of current Element Registry:" << std::endl;
  for (RegistryMap::const_iterator it=registry_.begin(); it != registry_.end(); ++it)
    os << it->first <<  " at address " << it->second << std::endl;
}			 
