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
//

// system include files
#include <cstring>
#include "tbb/concurrent_unordered_map.h"

// user include files
#include "FWCore/Utilities/interface/typelookup.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//
//This class hides the use of map from all classes that use HCTypeTag
namespace {
  struct StringEqual {
      bool operator()(const char* iLHS, const char* iRHS)   const {
         return (std::strcmp(iLHS, iRHS) == 0);
       }

   };
   
   //NOTE: The following hash calculation was taken from
   // the hash calculation for std::string in tbb

   //! A template to select either 32-bit or 64-bit constant as compile time, depending on machine word size.
   template <unsigned u, unsigned long long ull >
   struct select_size_t_constant {
      //Explicit cast is needed to avoid compiler warnings about possible truncation.
      //The value of the right size,   which is selected by ?:, is anyway not truncated or promoted.
      static constexpr size_t value = (size_t)((sizeof(size_t)==sizeof(u)) ? u : ull);
   };
   
   constexpr size_t hash_multiplier = select_size_t_constant<2654435769U, 11400714819323198485ULL>::value;

   struct StringHash {
      inline size_t operator()( const char* s) const {
         size_t h = 0;
         for( const char* c = s; *c; ++c )
            h = static_cast<size_t>(*c) ^ (h * hash_multiplier);
         return h;
      }
   };

   //NOTE: the use of const char* does not lead to a memory leak because the data
   // for the strings are assigned at compile time via a macro call
   using TypeNameToValueMap = tbb::concurrent_unordered_map<const char*, const std::type_info*, StringHash, StringEqual>;

   static TypeNameToValueMap& typeNameToValueMap() {
      static TypeNameToValueMap s_map;
      return s_map;
   }
}

edm::typelookup::NameRegistrar::NameRegistrar(const char* iTypeName,const std::type_info& iInfo)
{
   typeNameToValueMap().insert(std::pair<const char*, const std::type_info*>(iTypeName,&iInfo));
}


std::pair<const char*, const std::type_info*> 
edm::typelookup::findType(const char* iTypeName) {

   auto itFind = typeNameToValueMap().find(iTypeName);
   
   if(itFind == typeNameToValueMap().end()) {
      return std::make_pair(static_cast<const char*>(0), static_cast<std::type_info*> (0));
   }
   
   return (*itFind);
}
