#ifndef DataFormats_Common_fillPtrVector_h
#define DataFormats_Common_fillPtrVector_h
// -*- C++ -*-
//
// Package:     Common
// Class  :     fillPtrVector
//
/**\class fillPtrVector fillPtrVector.h DataFormats/Common/interface/fillPtrVector.h

 Description: Helper function used to implement the edm::Ptr class

 Usage:
    This is an internal detail of edm::Ptr interaction with edm::Wrapper and should not be used by others

*/
//
// Original Author:  Chris Jones
//         Created:  Sat Oct 20 11:45:38 CEST 2007
//

// system include files

// user include files
#include "DataFormats/Common/interface/FillView.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "Reflex/Object.h"
#include "Reflex/Type.h"

#include "DataFormats/Common/interface/fwd_fillPtrVector.h"

#include <typeinfo>
#include <vector>

// forward declarations
namespace edm {
  namespace detail {
    template <typename COLLECTION>
    void
    reallyfillPtrVector(COLLECTION const& coll,
                        std::type_info const& iToType,
                        std::vector<unsigned long> const& iIndicies,
                        std::vector<void const*>& oPtr)
    {
      typedef COLLECTION                            product_type;
      typedef typename GetProduct<product_type>::element_type     element_type;
      typedef typename product_type::const_iterator iter;
      typedef typename product_type::size_type      size_type;

      oPtr.reserve(iIndicies.size());
      if(iToType == typeid(element_type)) {
        for(std::vector<unsigned long>::const_iterator itIndex = iIndicies.begin(),
            itEnd = iIndicies.end();
            itIndex != itEnd;
            ++itIndex) {
          iter it = coll.begin();
          std::advance(it, *itIndex);
          element_type const* address = GetProduct<product_type>::address(it);
          oPtr.push_back(address);
        }
      } else {
        using Reflex::Type;
        using Reflex::Object;
        static Type const s_type(Type::ByTypeInfo(typeid(element_type)));
        Type toType = Type::ByTypeInfo(iToType);

        for(std::vector<unsigned long>::const_iterator itIndex = iIndicies.begin(),
            itEnd = iIndicies.end();
            itIndex != itEnd;
            ++itIndex) {
          iter it = coll.begin();
          std::advance(it, *itIndex);
          element_type const* address = GetProduct<product_type>::address(it);
          // The const_cast below is needed because
          // Object's constructor requires a pointer to
          // non-const void, although the implementation does not, of
          // course, modify the object to which the pointer points.
          Object obj(s_type, const_cast<void*>(static_cast<void const*>(address)));
          Object cast = obj.CastObject(toType);
          if(0 != cast.Address()) {
            oPtr.push_back(cast.Address());// returns void*, after pointer adjustment
          } else {
            Exception::throwThis(errors::LogicError,
            "TypeConversionError "
            "edm::PtrVector<> : unable to convert type ",
            typeid(element_type).name(),
            " to ",
            iToType.name(),
            "\n");
          }
        }
      }
    }
  }

  template <typename T, typename A>
  void
  fillPtrVector(std::vector<T, A> const& obj,
                std::type_info const& iToType,
                std::vector<unsigned long> const& iIndicies,
                std::vector<void const*>& oPtr) {
    detail::reallyfillPtrVector(obj, iToType, iIndicies, oPtr);
  }

  template <typename T, typename A>
  void
  fillPtrVector(std::list<T, A> const& obj,
                std::type_info const& iToType,
                std::vector<unsigned long> const& iIndicies,
                std::vector<void const*>& oPtr) {
    detail::reallyfillPtrVector(obj, iToType, iIndicies, oPtr);
  }

  template <typename T, typename A>
  void
  fillPtrVector(std::deque<T, A> const& obj,
                std::type_info const& iToType,
                std::vector<unsigned long> const& iIndicies,
                std::vector<void const*>& oPtr) {
    detail::reallyfillPtrVector(obj, iToType, iIndicies, oPtr);
  }

  template <typename T, typename A, typename Comp>
  void
  fillPtrVector(std::set<T, A, Comp> const& obj,
                std::type_info const& iToType,
                std::vector<unsigned long> const& iIndicies,
                std::vector<void const*>& oPtr) {
    detail::reallyfillPtrVector(obj, iToType, iIndicies, oPtr);
  }
}

#endif
