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

namespace std{} using namespace std;

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DetectorDescription/DDParser/interface/DDXMLElementRegistry.h"

// DDCore dependencies
#include "DetectorDescription/DDBase/interface/DDdebug.h"

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
  registry_ = new RegistryMap;
}

DDXMLElementRegistry::~DDXMLElementRegistry()
{
  delete registry_;
}

// -------------------------------------------------------------------------
// Implementation
// -------------------------------------------------------------------------

//  This allows Elements to register themselves with the static registry
void DDXMLElementRegistry::registerElement(const string& name, DDXMLElement* element)
{
  DCOUT_V('P',"DDXMLElementRegistry::registerElementBase: "  << name << " at " << element);
  (*registry_)[name] = element;
}

// THE most important part.  Getting the pointer to a given element type.
DDXMLElement* DDXMLElementRegistry::getElement(const string& name)
{
  DCOUT_V('P',  "DDXMLElementRegistry::getElement " << name << endl);
  RegistryMap::iterator it = registry_->find(name);
  DDXMLElement* myret = NULL;
  if (it != registry_->end())
    myret = it->second;
  DCOUT_V('P',  "DDXMLElementRegistry::getElement " << name << endl);
  return myret;
}

//  // Getting the iterator to a given element type.
//  DDXMLElementRegistry::RegistryMap::iterator DDXMLElementRegistry::find(const string& name) const
//  {
//    cout << "size of registry_ = " << registry_->size() << endl;
//    RegistryMap::iterator it = registry_->find(name);
//    DCOUT_V('P', "DDXMLElementRegistry::find(" << name << ") after setting iterator it." );
//    return it;
//  }

//  DDXMLElementRegistry::RegistryMap::iterator DDXMLElementRegistry::end() const
//  {
//    return registry_->end();
//  }

// Get the name given a pointer.  This may not be needed...
string DDXMLElementRegistry::getElementName(DDXMLElement* theElementBase)
{
  string ret = "";
  for (RegistryMap::const_iterator it = registry_->begin(); it != registry_->end(); it++)
    if (it->second == theElementBase)
      ret = it->first;
  return ret;
}

//  DDXMLElementRegistry::RegistryMap::iterator DDXMLElementRegistry::find(const string& name) const
//  {
//    return registry_->find(name);
//  }

//  DDXMLElementRegistry::RegistryMap::iterator DDXMLElementRegistry::end() const
//  {
//    return registry_->end();
//  }

ostream & operator<<(ostream & os, const DDXMLElementRegistry & element)
{
  element.stream(os);
  return os;
}

void DDXMLElementRegistry::stream(ostream & os) const
{
  os << "Output of current Element Registry:" << endl;
  for (RegistryMap::const_iterator it=registry_->begin(); it != registry_->end(); it++)
    os << it->first <<  " at address " << it->second << endl;
}			 
