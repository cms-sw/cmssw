#ifndef Framework_HCTypeTag_icc
#define Framework_HCTypeTag_icc
// -*- C++ -*-
//
// Package:     HeteroContainer
// Module:      HCTypeTag
// 
// Description: <one line class summary>
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris D. Jones
// Created:     Sun Sep 20 15:27:25 EDT 1998
// $Id: HCTypeTag.icc,v 1.6 2005/11/11 20:55:54 chrjones Exp $
//
// Revision history
//
// $Log: HCTypeTag.icc,v $
// Revision 1.6  2005/11/11 20:55:54  chrjones
// use new TypeIDBase for basis of all type comparisons
//
// Revision 1.5  2005/09/01 23:30:48  wmtan
// fix rule violations found by rulechecker
//
// Revision 1.4  2005/09/01 05:20:56  wmtan
// Fix Rules violations found by RuleChecker
//
// Revision 1.3  2005/07/14 22:50:52  wmtan
// Rename packages
//
// Revision 1.2  2005/06/23 19:59:30  wmtan
// fix rules violations found by rulechecker
//
// Revision 1.1  2005/05/29 02:29:53  wmtan
// initial population
//
// Revision 1.2  2005/04/04 20:31:22  chrjones
// added namespace
//
// Revision 1.1  2005/03/28 15:03:30  chrjones
// first submission
//
// Revision 1.2  2000/07/25 13:42:37  cdj
// HCTypeTag can now find a TypeTag from the name of a type
//
// Revision 1.1.1.1  1998/09/23 14:13:12  cdj
// first submission
//

// system include files
#include <map>
#include <cstring>

// user include files
//#include "Logger/interface/report.h"
#include "FWCore/Framework/interface/HCTypeTag.h"

// STL classes

//
// constants, enums and typedefs
//
namespace edm {
   namespace eventsetup {
      namespace heterocontainer {
         
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
            static std::map< const char*, edm::TypeIDBase, StringCompare>& typeNameToValueMap() {
               static std::map<const char*, edm::TypeIDBase,StringCompare> s_map;
               return s_map;
            }
         }
         
         //
         // constructors and destructor
         //
         //HCTypeTag::HCTypeTag()
         //{
         //}
         
         // HCTypeTag::HCTypeTag(const HCTypeTag& rhs)
         // {
         //    // do actual copying here; if you implemented
         //    // operator= correctly, you may be able to use just say      
         //    *this = rhs;
         // }
         
         //HCTypeTag::~HCTypeTag()
         //{
         //}
         
         //
         // assignment operators
         //
         // const HCTypeTag& HCTypeTag::operator=(const HCTypeTag& rhs)
         // {
         //   if(this != &rhs) {
         //      // do actual copying here, plus:
         //      // "SuperClass"::operator=(rhs);
         //   }
         //
         //   return *this;
         // }
         
         //
         // member functions
         //
         
         //
         // const member functions
         //
         
         //
         // static member functions
         //
         HCTypeTag
         HCTypeTag::findType(const std::string& iTypeName) {
            return HCTypeTag::findType(iTypeName.c_str());
         }
         
         HCTypeTag
         HCTypeTag::findType(const char* iTypeName)
         {
            std::map<const char*, TypeIDBase,StringCompare>::iterator itFind = typeNameToValueMap().find(iTypeName);
            
            if(itFind == typeNameToValueMap().end()) {
               return HCTypeTag();
            }
            
            return HCTypeTag((*itFind).second, (*itFind).first);
         }
         
         void
         HCTypeTag::registerName(const char* iTypeName, const std::type_info& iInfo)
         {
            typeNameToValueMap().insert(std::pair<const char*, TypeIDBase>(iTypeName,TypeIDBase(iInfo)));
         }
         
      }
   }
}

#endif
