// -*- C++ -*-
//
// Package:     Utilities
// Class  :     typelookup
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Wed Jan 20 14:26:49 CST 2010
// $Id: typelookup.cc,v 1.1 2010/01/23 02:00:15 chrjones Exp $
//

// system include files
#include <cstring>
#include <map>

// user include files
#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Utilities/interface/TypeIDBase.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//
//This class hides the use of map from all classes that use HCTypeTag
namespace {
   struct StringCompare {
      bool operator()(const char* iLHS, const char* iRHS) const {
         return (std::strcmp(iLHS, iRHS) < 0);
      }
   };
   
   //NOTE: the use of const char* does not lead to a memory leak because the data
   // for the strings are assigned at compile time via a macro call
   static std::map< const char*, const std::type_info*, StringCompare>& typeNameToValueMap() {
      static std::map<const char*, const std::type_info*,StringCompare> s_map;
      return s_map;
   }
}

edm::typelookup::NameRegistrar::NameRegistrar(const char* iTypeName,const std::type_info& iInfo)
{
   typeNameToValueMap().insert(std::pair<const char*, const std::type_info*>(iTypeName,&iInfo));
}


std::pair<const char*, const std::type_info*> 
edm::typelookup::findType(const char* iTypeName) {

   std::map<const char*, const std::type_info*,StringCompare>::iterator itFind = typeNameToValueMap().find(iTypeName);
   
   if(itFind == typeNameToValueMap().end()) {
      return std::make_pair(static_cast<const char*>(0), static_cast<std::type_info*> (0));
   }
   
   return (*itFind);
}
