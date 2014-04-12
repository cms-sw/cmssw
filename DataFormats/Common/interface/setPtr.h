#ifndef DataFormats_Common_setPtr_h
#define DataFormats_Common_setPtr_h
// -*- C++ -*-
//
// Package:     Common
// Class  :     setPtr
//
/**\class setPtr setPtr.h DataFormats/Common/interface/setPtr.h

 Description: Helper function used to implement the edm::Ptr class

 Usage:
    This is an internal detail of edm::Ptr interaction with edm::Wrapper and should not be used by others

*/
//
// Original Author:  Chris Jones
//         Created:  Sat Oct 20 11:45:38 CEST 2007
//

// user include files
#include "DataFormats/Common/interface/FillView.h"
#include "DataFormats/Common/interface/fwd_setPtr.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

// system include files
#include <typeinfo>
#include <vector>

// forward declarations
namespace edm {
  namespace detail {

    template <typename COLLECTION>
    void
    reallySetPtr(COLLECTION const& coll,
                 std::type_info const& iToType,
                 unsigned long iIndex,
                 void const*& oPtr) {
      typedef COLLECTION                            product_type;
      typedef typename GetProduct<product_type>::element_type     element_type;
      typedef typename product_type::const_iterator iter;
      typedef typename product_type::size_type      size_type;

      if(iToType == typeid(element_type)) {
        iter it = coll.begin();
        std::advance(it,iIndex);
        element_type const* address = GetProduct<product_type>::address( it );
        oPtr = address;
      } else {
        static TypeWithDict const s_type(TypeWithDict(typeid(element_type)));

        iter it = coll.begin();
        std::advance(it,iIndex);
        element_type const* address = GetProduct<product_type>::address( it );

        oPtr = TypeWithDict(iToType).pointerToBaseType(address, s_type);

        if(0 == oPtr) {
          Exception::throwThis(errors::LogicError,
            "TypeConversionError"
             "edm::Ptr<> : unable to convert type ",
             typeid(element_type).name(),
             " to ",
             iToType.name(),
             "\n");
        }
      }
    }
  }

  template <typename T, typename A>
  void
  setPtr(std::vector<T, A> const& obj,
         std::type_info const& iToType,
         unsigned long iIndex,
         void const*& oPtr) {
    detail::reallySetPtr(obj, iToType, iIndex, oPtr);
  }

  template <typename T, typename A>
  void
  setPtr(std::list<T, A> const& obj,
         std::type_info const& iToType,
         unsigned long iIndex,
         void const*& oPtr) {
    detail::reallySetPtr(obj, iToType, iIndex, oPtr);
  }

  template <typename T, typename A>
  void
  setPtr(std::deque<T, A> const& obj,
         std::type_info const& iToType,
         unsigned long iIndex,
         void const*& oPtr) {
    detail::reallySetPtr(obj, iToType, iIndex, oPtr);
  }

  template <typename T, typename A, typename Comp>
  void
  setPtr(std::set<T, A, Comp> const& obj,
         std::type_info const& iToType,
         unsigned long iIndex,
         void const*& oPtr) {
    detail::reallySetPtr(obj, iToType, iIndex, oPtr);
  }

}

#endif
