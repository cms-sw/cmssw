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
 the helper function es::products or 'output' the values to es::produced via the '<<' operator.

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
         struct Produce {
            Produce() {}
         };
         
         template< typename T> struct OneHolder {
            OneHolder() {}
            OneHolder(T iValue) : value_(std::move(iValue)) {}

            template<typename S>
            void setFromRecursive(S& iGiveValues) {
               iGiveValues.setFrom(value_);
            }
            
            void moveTo(T& oValue) { oValue = std::move(value_);}
            T value_;
            typedef T tail_type;
            typedef Null head_type;
         };

         template<typename T> struct OneHolder<std::unique_ptr<T>> {
            typedef std::unique_ptr<T> Type;
            OneHolder() {}
            OneHolder(OneHolder<T> const& ) = delete;
            OneHolder(OneHolder<Type>&& iOther): value_(std::move(iOther.value_)) {}
            OneHolder(Type iPtr): value_(std::move(iPtr)) {}
            
            
            OneHolder<Type> const& operator=(OneHolder<Type> iRHS) { value_ = std::move(iRHS.value_); return *this; }
            template<typename S>
            void setFromRecursive(S& iGiveValues) {
               iGiveValues.setFrom(value_);
            }
            
            void moveTo(Type& oValue) { oValue = std::move(value_);}
            mutable Type value_; //mutable needed for std::unique_ptr
            typedef Type tail_type;
            typedef Null head_type;
         };
         
        template<typename T> OneHolder<std::remove_reference_t<T>> operator<<(const Produce&, T&& iValue) {
            return OneHolder<std::remove_reference_t<T>>(std::forward<T>(iValue));
         }

         template<typename T, typename U> struct MultiHolder {
            MultiHolder(const T& iT, U iValue): value_(std::move(iValue)), head_(iT) {
            }
            
            template<typename TTaker>
            void setFromRecursive(TTaker& iGiveValues) {
               iGiveValues.setFrom(value_);
               head_.setFromRecursive(iGiveValues);
            }
            
            
            void moveTo(U& oValue) { oValue = std::move(value_);}
            U value_;
            T head_;
            
            typedef U tail_type;
            typedef T head_type;
         };

         template<typename T, typename S>
            MultiHolder<OneHolder<T>, std::remove_reference_t<S>> operator<<(OneHolder<T>&& iHolder,
                                                    S&& iValue) {
              return MultiHolder<OneHolder<T>, std::remove_reference_t<S>>(std::forward<OneHolder<T>>(iHolder), std::forward<S>(iValue));
            }
         template< typename T, typename U, typename V>
            MultiHolder< MultiHolder<T, U>, std::remove_reference_t<V> >
            operator<<(MultiHolder<T,U>&& iHolder, V&& iValue) {
              return MultiHolder< MultiHolder<T, U>, std::remove_reference_t<V>> (std::forward<MultiHolder<T,U>>(iHolder), std::forward<V>(iValue));
            }
        
         template<typename T1, typename... TArgs>
            struct ProductHolder : public ProductHolder<TArgs...> {
               typedef ProductHolder<TArgs...> parent_type;
              
              ProductHolder() : value() {}
               
               template<typename T>
                  void setAllValues(T& iValuesFrom) {
                     iValuesFrom.setFromRecursive(*this);
                  }
               using parent_type::moveTo;
               void moveTo(T1& oValue) { oValue = std::move(value); }

               using parent_type::setFrom;
               void setFrom(T1& iValue) { value = iValue ; }

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
               
               typedef T1 tail_type;
               typedef parent_type head_type;
            };
         
         template<typename T1>
            struct ProductHolder<T1> {
               
              ProductHolder() : value() {}

              template<typename T>
               void setAllValues(T& iValuesFrom) {
                  iValuesFrom.moveToRecursive(*this);
               }
               void moveTo(T1& oValue) { oValue = std::move(value); }
               void setFrom(T1& iValue) { value = iValue ; }
               template<typename T>
               void moveToRecursive(T& iValuesTo) {
                  iValuesTo.moveTo(value);
               }
               template<typename T>
               void setFromRecursive(T& iValuesTo) {
                  iValuesTo.setFrom(value);
               }
               T1 value;
               
               typedef T1 tail_type;
               typedef Null head_type;
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
      ESProducts(ESFillDirectly, Vars... vars) {
        (this->setFrom(vars), ...);
      }
     
   };

   namespace es {
      extern const eventsetup::produce::Produce produced;

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
