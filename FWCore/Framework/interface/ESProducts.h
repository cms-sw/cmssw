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
         struct Produce {};
         
         template< typename T> struct OneHolder {
            OneHolder() {}
            OneHolder(const T& iValue) : value_(iValue) {}

            template<typename S>
            void setFromRecursive(S& iGiveValues) {
               iGiveValues.setFrom(value_);
            }
            
            void assignTo(T& oValue) { oValue = value_;}
            T value_;
            typedef T tail_type;
            typedef Null head_type;
         };

         template< typename T> struct OneHolder< std::auto_ptr<T> > {
            typedef std::auto_ptr<T> Type;
            OneHolder() {}
            OneHolder(const OneHolder<Type>& iOther): value_(const_cast<OneHolder<Type>& >(iOther).value_) {}
            OneHolder(Type iPtr): value_(iPtr) {}
            
            
            const OneHolder<Type> & operator=(OneHolder<Type> iRHS) { value_ =iRHS.value_; return *this; }
            template<typename S>
            void setFromRecursive(S& iGiveValues) {
               iGiveValues.setFrom(value_);
            }
            
            void assignTo(Type& oValue) { oValue = value_;}
            mutable Type value_; //mutable needed for std::auto_ptr
            typedef Type tail_type;
            typedef Null head_type;
         };
         
         template<typename T> OneHolder<T> operator<<(const Produce&, T iValue) {
            return OneHolder<T>(iValue);
         }

         template<typename T, typename U> struct MultiHolder {
            MultiHolder(const T& iT, U iValue): value_(iValue), head_(iT) {
            }
            
            template<typename TTaker>
            void setFromRecursive(TTaker& iGiveValues) {
               iGiveValues.setFrom(value_);
               head_.setFromRecursive(iGiveValues);
            }
            
            
            void assignTo(U& oValue) { oValue = value_;}
            U value_;
            T head_;
            
            typedef U tail_type;
            typedef T head_type;
         };

         template<typename T, typename S>
            MultiHolder<OneHolder<T>, S> operator<<(const OneHolder<T>& iHolder,
                                                     const S& iValue) {
               return MultiHolder<OneHolder<T>, S>(iHolder, iValue);
            }
         template< typename T, typename U, typename V>
            MultiHolder< MultiHolder<T, U>, V >
            operator<<(const MultiHolder<T,U>& iHolder, const V& iValue) {
               return MultiHolder< MultiHolder<T, U>, V> (iHolder, iValue);
            }
         
         template<typename T1, typename T2, typename T3> 
            struct ProductHolder : public ProductHolder<T2, T3, Null> {
               typedef ProductHolder<T2, T3, Null> parent_type;
              
              ProductHolder() : value() {}
               
               template<typename T>
                  void setAllValues(T& iValuesFrom) {
                     iValuesFrom.setFromRecursive(*this);
                  }
               using parent_type::assignTo;
               void assignTo(T1& oValue) { oValue = value; }

               using parent_type::setFrom;
               void setFrom(T1& iValue) { value = iValue ; }

               template<typename T>
                  void setFromRecursive(T& iValuesTo) {
                     iValuesTo.setFrom(value);
                     parent_type::setFromRecursive(iValuesTo);
                  }

               template<typename T>
                  void assignToRecursive(T& iValuesTo) {
                     iValuesTo.assignTo(value);
                     parent_type::assignToRecursive(iValuesTo);
                  }
               T1 value;
               
               typedef T1 tail_type;
               typedef parent_type head_type;
            };
         
         template<typename T1>
            struct ProductHolder<T1, Null, Null> {
               
              ProductHolder() : value() {}

              template<typename T>
               void setAllValues(T& iValuesFrom) {
                  iValuesFrom.assignToRecursive(*this);
               }
               void assignTo(T1& oValue) { oValue = value; }
               void setFrom(T1& iValue) { value = iValue ; }
               template<typename T>
               void assignToRecursive(T& iValuesTo) {
                  iValuesTo.assignTo(value);
               }
               template<typename T>
               void setFromRecursive(T& iValuesTo) {
                  iValuesTo.setFrom(value);
               }
               T1 value;
               
               typedef T1 tail_type;
               typedef Null head_type;
            };
         
         template<>
            struct ProductHolder<Null, Null, Null > {
               void setAllValues(Null&) {}
               void assignTo(void*) {}
               void setFrom(void*) {}
               
               typedef Null tail_type;
               typedef Null head_type;
            };
      }
   }
   template<typename T1, typename T2 = eventsetup::produce::Null, typename T3 = eventsetup::produce::Null >
   struct ESProducts : public eventsetup::produce::ProductHolder<T1, T2, T3> {
      template<typename S1, typename S2, typename S3>
      ESProducts(const ESProducts<S1, S2, S3>& iProducts) {
         setAllValues(const_cast<ESProducts<S1, S2, S3>&>(iProducts));
      }
      template<typename T>
      /*explicit*/ ESProducts(const T& iValues) {
         setAllValues(const_cast<T&>(iValues));
      }
   };

   namespace es {
      extern eventsetup::produce::Produce produced;

      template<typename T, typename S>
      ESProducts<T,S> products(const T& i1, const S& i2) {
         return ESProducts<T,S>(produced << i1 << i2);
      }

      template<typename T, typename S, typename U>
         ESProducts<T,S, U> products(const T& i1, const S& i2, const U& i3) {
            return ESProducts<T,S,U>(produced << i1 << i2 << i3);
         }
   }

   template<typename T1, typename T2, typename T3, typename ToT> 
     void copyFromTo(ESProducts<T1,T2,T3>& iFrom,
                     ToT& iTo) {
       iFrom.assignTo(iTo);
     }
}


#endif
