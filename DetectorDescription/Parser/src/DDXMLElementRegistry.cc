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

#include <algorithm>
#include <vector>
#include <iostream>

// // -------------------------------------------------------------------------
// // Constructor/Destructor
// // -------------------------------------------------------------------------

// DDXMLElementRegistry::DDXMLElementRegistry()
// {
//   //  registry_ = new RegistryMap;
// }


// -------------------------------------------------------------------------
// Implementation
// -------------------------------------------------------------------------

// //  This allows Elements to register themselves with the static registry
// void DDXMLElementRegistry::registerElement(const std::string& name, DDXMLElement* element)
// {
//   DCOUT_V('P',"DDXMLElementRegistry::registerElementBase: "  << name << " at " << element);
//   registry_[name] = element;
// }

// THE most important part.  Getting the pointer to a given element type.
// DD
// std::ostream & operator<<(std::ostream & os, const DDXMLElementRegistry & element)
// {
//   element.stream(os);
//   return os;
// }

// void DDXMLElementRegistry::stream(std::ostream & os) const
// {
//   os << "Output of current Element Registry:" << std::endl;
//   for (RegistryMap::const_iterator it=registry_.begin(); it != registry_.end(); ++it)
//     os << it->first <<  " at address " << it->second << std::endl;
// }			 
