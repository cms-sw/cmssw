#ifndef FWCore_Utilities_TypeToGet_h
#define FWCore_Utilities_TypeToGet_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     TypeToGet
// 
/**\class TypeToGet TypeToGet.h "FWCore/Utilities/interface/TypeToGet.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sat, 06 Apr 2013 14:45:01 GMT
// $Id: TypeToGet.h,v 1.1 2013/04/14 19:01:14 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"

// forward declarations
namespace edm {
  template< typename T> class View;

  class TypeToGet
  {
    
  public:
    
    /**If the type is is a edm::View<T> then
     iID should be typeid(T) and iKind should be
     edm::ELEMENT_TYPE. Else TypeID should be the full type
     and iKind should be edm::ProductType **/
    TypeToGet(TypeID const& iID, KindOfType iKind):
    m_type(iID), m_kind(iKind) {}

    // ---------- const member functions ---------------------
    TypeID const& type() const {return m_type;}
    KindOfType kind() const {return m_kind;}
    
    // ---------- static member functions --------------------
    template<typename T>
    static TypeToGet make() {
      return TypeToGet(edm::TypeID(typeIdFor(static_cast<T*>(nullptr))),kindOfTypeFor(static_cast<T*>(nullptr)));
    }

  private:

    template<typename T>
    static std::type_info const& typeIdFor(T*) { return typeid(T); }

    template<typename T>
    static std::type_info const& typeIdFor(edm::View<T>*) { return typeid(T); }

    template<typename T>
    static KindOfType kindOfTypeFor(T*) { return PRODUCT_TYPE; }
    
    template<typename T>
    static KindOfType kindOfTypeFor(edm::View<T>*) { return ELEMENT_TYPE; }

    
    TypeToGet() = delete;
    
    // ---------- member data --------------------------------
    TypeID m_type;
    KindOfType m_kind;
  };
}

#endif
