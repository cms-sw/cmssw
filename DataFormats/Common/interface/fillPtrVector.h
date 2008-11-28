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
// $Id: fillPtrVector.h,v 1.3 2008/04/22 22:17:35 wmtan Exp $
//

// system include files

// user include files
#include "DataFormats/Common/interface/FillView.h"
#include "Reflex/Object.h"
#include "Reflex/Type.h"

// forward declarations
namespace edm {
  namespace detail {
    
    
    template <class COLLECTION>
    void
    reallyfillPtrVector(COLLECTION const& coll,
                        const std::type_info& iToType,
                        const std::vector<unsigned long>& iIndicies,
                        std::vector<void const*>& oPtr)
    {
      typedef COLLECTION                            product_type;
      typedef typename GetProduct<product_type>::element_type     element_type;
      typedef typename product_type::const_iterator iter;
      typedef typename product_type::size_type      size_type;
      
      oPtr.reserve(iIndicies.size());
      if(iToType == typeid(element_type)) {
        for(std::vector<unsigned long>::const_iterator itIndex=iIndicies.begin(),
            itEnd = iIndicies.end();
            itIndex != itEnd;
            ++itIndex) {
          iter it = coll.begin();
          advance(it,*itIndex);          
          element_type const* address = GetProduct<product_type>::address( it );
          oPtr.push_back(address);
        }
      } else {
        using Reflex::Type;
        using Reflex::Object;
        static const Type s_type(Type::ByTypeInfo(typeid(element_type)));
        Type toType=Type::ByTypeInfo(iToType);
        
        for(std::vector<unsigned long>::const_iterator itIndex=iIndicies.begin(),
            itEnd = iIndicies.end();
            itIndex != itEnd;
            ++itIndex) {
          iter it = coll.begin();
          advance(it,*itIndex);          
          element_type const* address = GetProduct<product_type>::address( it );
          // The const_cast below is needed because
          // Object's constructor requires a pointer to
          // non-const void, although the implementation does not, of
          // course, modify the object to which the pointer points.
          Object obj(s_type, const_cast<void*>(static_cast<const void*>(address)));
          Object cast = obj.CastObject(toType);
          if(0 != cast.Address()) {
            oPtr.push_back(cast.Address());// returns void*, after pointer adjustment
          } else {
            throw cms::Exception("TypeConversionError")
            << "edm::PtrVector<> : unable to convert type " << typeid(element_type).name()
            << " to " << iToType.name() << "\n";
          }
          
        }
      }
    }
  }
  
  template <class T, class A>
  void
  fillPtrVector(std::vector<T,A> const& obj,
                const std::type_info& iToType,
                const std::vector<unsigned long>& iIndicies,
                std::vector<void const*>& oPtr) {
    detail::reallyfillPtrVector(obj, iToType, iIndicies, oPtr);
  }
  
  template <class T, class A>
  void
  fillPtrVector(std::list<T,A> const& obj,
                const std::type_info& iToType,
                const std::vector<unsigned long>& iIndicies,
                std::vector<void const*>& oPtr) {
    detail::reallyfillPtrVector(obj, iToType, iIndicies, oPtr);
  }
  
  template <class T, class A>
  void
  fillPtrVector(std::deque<T,A> const& obj,
                const std::type_info& iToType,
                const std::vector<unsigned long>& iIndicies,
                std::vector<void const*>& oPtr) {
    detail::reallyfillPtrVector(obj, iToType, iIndicies, oPtr);
  }
  
  template <class T, class A, class Comp>
  void
  fillPtrVector(std::set<T,A,Comp> const& obj,
                const std::type_info& iToType,
                const std::vector<unsigned long>& iIndicies,
                std::vector<void const*>& oPtr) {
    detail::reallyfillPtrVector(obj, iToType, iIndicies, oPtr);
  }

}

#endif
