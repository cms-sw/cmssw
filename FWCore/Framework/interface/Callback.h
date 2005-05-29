#ifndef EVENTSETUP_CALLBACK_H
#define EVENTSETUP_CALLBACK_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     Callback
// 
/**\class Callback Callback.h Core/CoreFramework/interface/Callback.h

 Description: Functional object used as the 'callback' for the CallbackProxy

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Sun Apr 17 14:30:24 EDT 2005
// $Id: Callback.h,v 1.1 2005/04/18 20:16:16 chrjones Exp $
//

// system include files
#include <vector>
// user include files
#include "FWCore/CoreFramework/interface/produce_helpers.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      
      //need a virtual distructor since owner of callback only knows
      // about the base class.  Other users of callback know the specific
      // type
      struct CallbackBase { virtual ~CallbackBase() {} };
      
      template<typename T, typename TReturn, typename TRecord>
      class Callback : public CallbackBase {
       public:
         typedef TReturn (T ::* method_type)(const TRecord& );
         
         Callback( T* iProd, method_type iMethod ) :
            proxyData_(produce::size< TReturn >::value, static_cast<void*>(0) ),
            producer_(iProd), method_(iMethod) {}
         
         
         // ---------- const member functions ---------------------
         
         // ---------- static member functions --------------------
         
         // ---------- member functions ---------------------------
         
         
         void operator()(const TRecord& iRecord ) { 
            if( !wasCalledForThisRecord_ ) {
               storeReturnedValues( (producer_->*method_)( iRecord ) );                  
               wasCalledForThisRecord_ = true;
            }
         }
         
         template<class DataT>
            void holdOntoPointer(DataT* iData) {
               proxyData_[ produce::find_index<TReturn,DataT>::value ] = iData;
            }
         
         void storeReturnedValues( TReturn iReturn ) {
            //std::cout <<" storeReturnedValues "<< iReturn <<" " <<iReturn->value_ <<std::endl;
            typedef typename produce::product_traits<TReturn>::type type;
            setData( iReturn, static_cast<typename  type::head_type*>(0), static_cast<const typename type::tail_type *>(0) );
         }
         
         template<class RemainingContainerT, class DataT, class ProductsT>
            void setData( ProductsT& iProducts, const RemainingContainerT*, const DataT* ) {
               DataT* temp = reinterpret_cast< DataT*>( proxyData_[produce::find_index<TReturn,DataT>::value] ) ;
               if( 0 != temp ) { produce::copyFromTo(iProducts, *temp); }
               setData(iProducts, static_cast< const typename RemainingContainerT::head_type *>(0),
                       static_cast< const typename RemainingContainerT::tail_type *>(0) );
            }
         template<class DataT, class ProductsT>
            void setData(ProductsT& iProducts, const produce::Null*, const DataT* ) {
               
               DataT* temp = reinterpret_cast< DataT*>( proxyData_[produce::find_index<TReturn,DataT>::value] ) ;
               //std::cout <<" setData["<< produce::find_index<TReturn,DataT>::value<<"] "<< temp <<std::endl;
               if( 0 != temp ) { produce::copyFromTo(iProducts, *temp); } 
            };
         void newRecordComing() {
            wasCalledForThisRecord_ = false;
         }
         
     private:
         Callback( const Callback& ); // stop default
         
         const Callback& operator=( const Callback& ); // stop default

         std::vector<void*> proxyData_;
         T* producer_;
         method_type method_;
         bool wasCalledForThisRecord_;
      };
   }
}

#endif /* EVENTSETUP_CALLBACK_H */
