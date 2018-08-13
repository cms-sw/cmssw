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
            using tail_type = T;
            using head_type = Null;
         };
         template <typename T> struct product_traits {
            using type = T;
         };
         template< typename T> struct product_traits<T*> {
            using type = EndList<T*>;
         };
         template< typename T> struct product_traits<std::unique_ptr<T> > {
            using type= EndList<std::unique_ptr<T>>;
         };
         template< typename T> struct product_traits<std::shared_ptr<T> > {
            using type = EndList<std::shared_ptr<T>>;
         };
         template< typename T> struct product_traits<std::optional<T> > {
            using type=EndList<std::optional<T>>;
         };
         
         template<typename T> struct size {
            using type = typename product_traits<T>::type;
            constexpr static int value = size< typename type::head_type >::value + 1;
         };
         template<> struct size<Null> {
            constexpr static int value = 0;
         };
         
         template<typename T> struct smart_pointer_traits {
            using type = typename T::element_type;
           static auto getPointer(T& iPtr)-> decltype(&*iPtr) { return &*iPtr;}
         };
       
         template<typename T> struct smart_pointer_traits<std::optional<T>> {
           using type = T;
           static T* getPointer(std::optional<T>& iPtr) {
             if(iPtr.has_value()) { return &*iPtr;}
             return nullptr;
           }
         };

         template<typename T, typename FindT> struct find_index {
            using container_type = typename product_traits<T>::type;
            template<typename HeadT, typename TailT>
            constexpr static int findIndexOf() {
              if constexpr(not std::is_same_v<TailT,FindT>) {
                using container_type= typename product_traits<HeadT>::type;
                return findIndexOf<typename container_type::head_type,
                typename container_type::tail_type>()+1;
              } else {
                return 0;
              }
            }
            constexpr static int value = findIndexOf<typename container_type::head_type, typename container_type::tail_type>();
         };
         namespace test {
            template<typename T> const char* name(const T*);
         }
         
      }
   }
}

#endif
