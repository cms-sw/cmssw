#ifndef FWCore_Framework_ESProducts_h
#define FWCore_Framework_ESProducts_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESProducts
// 
/**\class ESProducts ESProducts.h FWCore/Framework/interface/ESProducts.h

 Description: Container for multiple products created by an ESProducer

 Usage:
    This is used as a return type from the produce method of an ESProducer.  Users are not anticipated to ever need
 to directly use this class.  Instead, to create an instance of the class at method return time, they should call
 the helper function es::products.

*/
//
// Author:      Chris Jones
// Created:     Sun Apr 17 17:30:46 EDT 2005
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/produce_helpers.h"

// forward declarations
namespace edm {
   namespace eventsetup {      
      namespace produce {
         template<typename T1, typename... TArgs>
            struct ProductHolder : public ProductHolder<TArgs...> {
              using parent_type = ProductHolder<TArgs...>;
              
              ProductHolder() : value() {}
              ProductHolder(ProductHolder<T1,TArgs...>&&) = default;
              ProductHolder(ProductHolder<T1,TArgs...> const&) = default;
              ProductHolder<T1, TArgs...>& operator=(ProductHolder<T1,TArgs...>&&) = default;
              ProductHolder<T1, TArgs...>& operator=(ProductHolder<T1,TArgs...> const&) = default;

               template<typename T>
                  void setAllValues(T& iValuesFrom) {
                     iValuesFrom.setFromRecursive(*this);
                  }
               using parent_type::moveTo;
               void moveTo(T1& oValue) { oValue = std::move(value); }

               using parent_type::setFrom;
               void setFrom(T1& iValue) { value = iValue ; }
               void setFrom(T1&& iValue) { value = std::move(iValue) ; }

               template<typename T>
                  void setFromRecursive(T& iValuesTo) {
                     iValuesTo.setFrom(value);
                     parent_type::setFromRecursive(iValuesTo);
                  }

               template<typename T>
                  void moveToRecursive(T& iValuesTo) {
                     iValuesTo.moveTo(value);
                     parent_type::moveToRecursive(iValuesTo);
                  }
               T1 value;
               
               using tail_type = T1;
               using head_type = parent_type;
            };
         
         template<typename T1>
            struct ProductHolder<T1> {
               
              ProductHolder() : value() {}
              ProductHolder(ProductHolder<T1>&&) = default;
              ProductHolder(ProductHolder<T1> const&) = default;
              ProductHolder<T1>& operator=(ProductHolder<T1>&&) = default;
              ProductHolder<T1>& operator=(ProductHolder<T1> const&) = default;

              template<typename T>
               void setAllValues(T& iValuesFrom) {
                  iValuesFrom.moveToRecursive(*this);
               }
               void moveTo(T1& oValue) { oValue = std::move(value); }
               void setFrom(T1& iValue) { value = iValue ; }
               void setFrom(T1&& iValue) { value = std::move(iValue) ; }
               template<typename T>
               void moveToRecursive(T& iValuesTo) {
                  iValuesTo.moveTo(value);
               }
               template<typename T>
               void setFromRecursive(T& iValuesTo) {
                  iValuesTo.setFrom(value);
               }
               T1 value;
               
               using tail_type = T1;
               using head_type = Null;
            };
         
      }
   }
   struct ESFillDirectly {};

   template<typename ...TArgs>
   struct ESProducts : public eventsetup::produce::ProductHolder<TArgs...> {
      typedef eventsetup::produce::ProductHolder<TArgs...> parent_type;
      template<typename... S>
      ESProducts(ESProducts<S...>&& iProducts) {
         parent_type::setAllValues(iProducts);
      }
      template<typename T>
      /*explicit*/ ESProducts(T&& iValues) {
         parent_type::setAllValues(iValues);
      }
      template<typename ...Vars>
      ESProducts(ESFillDirectly, Vars&&... vars) {
         (this->setFrom(std::forward<Vars>(vars)), ...);
      }

      ESProducts(ESProducts<TArgs...> const&) = default;
      ESProducts(ESProducts<TArgs...>&&) = default;
      ESProducts<TArgs...>& operator=(ESProducts<TArgs...> const&) = default;
      ESProducts<TArgs...>& operator=(ESProducts<TArgs...>&&) = default;
   };

   namespace es {
      template<typename ...TArgs>
      ESProducts<std::remove_reference_t<TArgs>...> products(TArgs&&... args) {
         return ESProducts<std::remove_reference_t<TArgs>...>(edm::ESFillDirectly{}, std::forward<TArgs>(args)...);
      }
   }

   template<typename ...TArgs, typename ToT>
     void moveFromTo(ESProducts<TArgs...>& iFrom,
                     ToT& iTo) {
       iFrom.moveTo(iTo);
     }
}


#endif
