#ifndef Framework_produce_helpers_h
#define Framework_produce_helpers_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     produce_helpers
// 
/**\class produce_helpers produce_helpers.h FWCore/Framework/interface/produce_helpers.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Fri Apr 15 10:25:20 EDT 2005
//

// system include files
#include <memory>
#include <optional>
// user include files

// forward declarations
namespace edm {
   namespace eventsetup {
      
      namespace produce { struct Null;}
      
     template<typename FromT, typename ToT> void moveFromTo(FromT& iFrom,
                                                            ToT & iTo) {
       iTo = std::move(iFrom);
     }

     template<typename FromT, typename ToT> void moveFromTo(std::unique_ptr<FromT>& iFrom, ToT & iTo) {
       iTo = std::move(iFrom);
     }
     template<typename FromT, typename ToT> void moveFromTo(std::optional<FromT>& iFrom, ToT & iTo) {
       iTo = std::move(iFrom.value());
     }

     namespace produce { 
         struct Null {};
         template <typename T> struct EndList {
           static_assert( (not std::is_pointer_v<T>) ,"use std::shared_ptr or std::unique_ptr to hold EventSetup data products, do not use bare pointers");
            typedef T tail_type;
            typedef Null head_type;
         };
         template <typename T> struct product_traits {
            typedef T type;
         };
         template< typename T> struct product_traits<T*> {
            typedef EndList<T*> type;
         };
         template< typename T> struct product_traits<std::unique_ptr<T> > {
            typedef EndList<std::unique_ptr<T> > type;
         };
         template< typename T> struct product_traits<std::shared_ptr<T> > {
            typedef EndList<std::shared_ptr<T> > type;
         };
         template< typename T> struct product_traits<std::optional<T> > {
            using type=EndList<std::optional<T>>;
         };
         
         template<typename T> struct size {
            typedef typename product_traits<T>::type type;
            enum { value = size< typename type::head_type >::value + 1 };
         };
         template<> struct size<Null> {
            enum { value = 0 };
         };
         
         template<typename T> struct smart_pointer_traits {
            typedef typename T::element_type type;
           static auto getPointer(T& iPtr)-> decltype(&*iPtr) { return &*iPtr;}
         };
       
         template<typename T> struct smart_pointer_traits<std::optional<T>> {
           using type = T;
           static T* getPointer(std::optional<T>& iPtr) {
             if(iPtr.has_value()) { return &*iPtr;}
             return nullptr;
           }
         };

         template<typename FromT, typename ToT> void moveFromTo(FromT& iFrom,
                                                            ToT & iTo) {
           iTo = std::move(iFrom);
         }

         template<typename FromT, typename ToT> void moveFromTo(std::unique_ptr<FromT>& iFrom, ToT & iTo) {
           iTo = std::move(iFrom);
         }


         template<typename ContainerT, typename EntryT, typename FindT> struct find_index_impl {
            typedef typename product_traits<ContainerT>::type container_type;
            enum { value = find_index_impl<typename container_type::head_type, typename container_type::tail_type,  FindT>::value + 1 };
         };
         template<typename ContainerT, typename T> struct find_index_impl<ContainerT, T,T> {
            enum { value = 0 };
         };
         
         template<typename T, typename FindT> struct find_index {
            typedef typename product_traits<T>::type container_type;
            enum {value = find_index_impl<typename container_type::head_type, typename container_type::tail_type, FindT>::value };
         };
         namespace test {
            template<typename T> const char* name(const T*);
         }
         
      }
   }
}

#endif
