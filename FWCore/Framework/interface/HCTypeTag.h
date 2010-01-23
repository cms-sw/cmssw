#ifndef Framework_HCTypeTag_h
#define Framework_HCTypeTag_h
// -*- C++ -*-
//
// Package:     HeteroContainer
// Module:      HCTypeTag
// 
// Description: Base class for Tags that can specify a class Type.
//
// Usage:
//    This class is used to specify the class Type of another class.
//    To build a TypeTag you must call the method TypeTag::make<T>()
//
//    Example
//       //Make a TypeTag for class Foo
//       HCTypeTag fooTag = HCTypeTag::make< Foo>(); 
//
// Author:      Chris D. Jones
// Created:     Sun Sep 20 15:05:10 EDT 1998
// $Id: HCTypeTag.h,v 1.9 2010/01/21 15:44:11 chrjones Exp $
//
//

// system include files
#include <string>

// user include files
#include "FWCore/Utilities/interface/TypeIDBase.h"
#include "FWCore/Utilities/interface/typelookup.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      namespace heterocontainer {
         
         using typelookup::className;
         
         class HCTypeTag : public TypeIDBase
         {
            // ---------- friend classes and functions ---------------
         public:
            // ---------- constants, enums and typedefs --------------
            
            // ---------- Constructors and destructor ----------------
            HCTypeTag() : m_name("") {}
            //virtual ~HCTypeTag();  
            
            // ---------- member functions ---------------------------
            
            // ---------- const member functions ---------------------
            const std::type_info& value() const { return typeInfo(); }
            const char*  name() const { return m_name; }
            
            ///find a type based on the types name, if not found will return default HCTypeTag
            static HCTypeTag findType(const char* iTypeName);
            static HCTypeTag findType(const std::string& iTypeName);
            
            template <typename T>
            static HCTypeTag make() {
               return HCTypeTag(typelookup::classTypeInfo<T>(),typelookup::className<T>());
            }
            
         protected:
            // ---------- protected member functions -----------------
            HCTypeTag(const std::type_info& iValue, const char* iName) :
            TypeIDBase(iValue), m_name(iName) {}
            
            HCTypeTag(const TypeIDBase& iValue, const char* iName) :
            TypeIDBase(iValue), m_name(iName) {}
            
            // ---------- protected const member functions -----------
            
            // ---------- protected static member functions ----------
            
         private:
            // ---------- Constructors and destructor ----------------
            //HCTypeTag(const HCTypeTag&); // use default
            
            // ---------- assignment operator(s) ---------------------
            //const HCTypeTag& operator=(const HCTypeTag&); // use default
            
            
            // ---------- data members -------------------------------
            const char*  m_name;
            
            
         };         
      }
   }
}
#define HCTYPETAG_HELPER_METHODS(_dataclass_) TYPELOOKUP_METHODS(_dataclass_)

#define DEFINE_HCTYPETAG_REGISTRATION(type) DEFINE_TYPELOOKUP_REGISTRATION(type)
#endif
