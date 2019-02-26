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
//
//

// user include files
#include "FWCore/Utilities/interface/TypeIDBase.h"
#include "FWCore/Utilities/interface/typelookup.h"

// system include files
#include <string>
#include <typeinfo>

// forward declarations
namespace edm {
  namespace eventsetup {
    namespace heterocontainer {

      using typelookup::className;

      class HCTypeTag : public TypeIDBase {
      public:

        HCTypeTag() = default;

        // ---------- member functions ---------------------------

        // ---------- const member functions ---------------------
        std::type_info const& value() const { return typeInfo(); }
        char const* name() const { return m_name; }

        ///find a type based on the types name, if not found will return default HCTypeTag
        static HCTypeTag findType(char const* iTypeName);
        static HCTypeTag findType(std::string const& iTypeName);

        template <typename T>
        static HCTypeTag make() {
          return HCTypeTag(typelookup::classTypeInfo<T>(),typelookup::className<T>());
        }

      protected:
        // ---------- protected member functions -----------------
        HCTypeTag(std::type_info const& iValue, char const* iName) :
          TypeIDBase(iValue), m_name(iName) {}

        HCTypeTag(TypeIDBase const& iValue, const char* iName) :
          TypeIDBase(iValue), m_name(iName) {}

      private:
        char const* m_name{""};
      };
    }
  }
}
#define HCTYPETAG_HELPER_METHODS(_dataclass_) TYPELOOKUP_METHODS(_dataclass_)

#define DEFINE_HCTYPETAG_REGISTRATION(type) DEFINE_TYPELOOKUP_REGISTRATION(type)
#endif
