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
// $Id: HCTypeTag.h,v 1.7 2009/04/26 22:18:35 chrjones Exp $
//
//

// system include files
#include <string>

// user include files
#include "FWCore/Utilities/interface/TypeIDBase.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      namespace heterocontainer {
         class HCTypeTagRegistrar {
         public:
            HCTypeTagRegistrar(const char* iTypeName,const std::type_info& iInfo);
         };
         
         template <typename T>
         const char* className();
         
         class HCTypeTag : public TypeIDBase
         {
            // ---------- friend classes and functions ---------------
            friend class HCTypeTagRegistrar;
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
               return HCTypeTag(typeid(T),className<T>());
            }
            
         protected:
            // ---------- protected member functions -----------------
            HCTypeTag(const std::type_info& iValue, const char* iName) :
            TypeIDBase(iValue), m_name(iName) {}
            
            HCTypeTag(const TypeIDBase& iValue, const char* iName) :
            TypeIDBase(iValue), m_name(iName) {}
            
            // ---------- protected const member functions -----------
            
            // ---------- protected static member functions ----------
            static void registerName(const char* iTypeName,const std::type_info& iInfo);
            
         private:
            // ---------- Constructors and destructor ----------------
            //HCTypeTag(const HCTypeTag&); // use default
            
            // ---------- assignment operator(s) ---------------------
            //const HCTypeTag& operator=(const HCTypeTag&); // use default
            
            
            // ---------- data members -------------------------------
            const char*  m_name;
            
            
         };
         
         inline HCTypeTagRegistrar::HCTypeTagRegistrar(const char* iTypeName,const std::type_info& iInfo) {
            HCTypeTag::registerName(iTypeName,iInfo);
         }
         
      }
   }
}
#define HCTYPETAG_CLASSNAME(Tname) \
template \
const char* \
edm::eventsetup::heterocontainer::className< Tname >() \
{ return #Tname ; }

#define HCTYPETAG_CLASSNAME_1_COMMA(Tname1, Tname2) \
template \
const char* \
edm::eventsetup::heterocontainer::className< Tname1,Tname2 >() \
{ return #Tname1 "," #Tname2 ; }

#define EDM_HCTYPETAG_SYM(x,y) EDM_HCTYPETAG_SYM2(x,y)
#define EDM_HCTYPETAG_SYM2(x,y) x ## y

#define DEFINE_HCTYPETAG_REGISTRATION(type) \
static edm::eventsetup::heterocontainer::HCTypeTagRegistrar EDM_HCTYPETAG_SYM(s_register , __LINE__ ) (edm::eventsetup::heterocontainer::className<type>(),typeid(type))

#endif
