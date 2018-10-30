#ifndef Framework_Callback_h
#define Framework_Callback_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     Callback
// 
/**\class Callback Callback.h FWCore/Framework/interface/Callback.h

 Description: Functional object used as the 'callback' for the CallbackProxy

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Sun Apr 17 14:30:24 EDT 2005
//

// system include files
#include <vector>
#include <type_traits>
// user include files
#include "FWCore/Framework/interface/produce_helpers.h"
#include "FWCore/Utilities/interface/propagate_const.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      
      //need a virtual distructor since owner of callback only knows
      // about the base class.  Other users of callback know the specific
      // type
      struct CallbackBase { virtual ~CallbackBase() {} };
      
      // The default decorator that does nothing
      template< typename TRecord>
      struct CallbackSimpleDecorator {
         void pre(const TRecord&) {}
         void post(const TRecord&) {}
      };
      
      template<typename T,         //producer's type
               typename TReturn,   //return type of the producer's method
               typename TRecord,   //the record passed in as an argument
               typename TDecorator //allows customization using pre/post calls 
                             =CallbackSimpleDecorator<TRecord> >
      class Callback : public CallbackBase {
       public:
         using  method_type = TReturn (T ::*)(const TRecord&);
         
         Callback(T* iProd, 
                   method_type iMethod,
                   const TDecorator& iDec = TDecorator()) :
            proxyData_{},
            producer_(iProd), 
            method_(iMethod),
            wasCalledForThisRecord_(false),
            decorator_(iDec) {}
         
         
         // ---------- const member functions ---------------------
         
         // ---------- static member functions --------------------
         
         // ---------- member functions ---------------------------
         
         
         void operator()(const TRecord& iRecord) { 
            if(!wasCalledForThisRecord_) {
               decorator_.pre(iRecord);
               storeReturnedValues((producer_->*method_)(iRecord));                  
               wasCalledForThisRecord_ = true;
               decorator_.post(iRecord);
            }
         }
         
         template<class DataT>
            void holdOntoPointer(DataT* iData) {
               proxyData_[produce::find_index<TReturn,DataT>::value] = iData;
            }
         
         void storeReturnedValues(TReturn iReturn) {
            //std::cout <<" storeReturnedValues "<< iReturn <<" " <<iReturn->value_ <<std::endl;
            using type = typename produce::product_traits<TReturn>::type;
            setData<typename type::head_type, typename type::tail_type>(iReturn);
         }
         
         template<class RemainingContainerT, class DataT, class ProductsT>
            void setData(ProductsT& iProducts) {
               DataT* temp = reinterpret_cast< DataT*>(proxyData_[produce::find_index<TReturn,DataT>::value]) ;
               if(nullptr != temp) { moveFromTo(iProducts, *temp); }
               if constexpr( not std::is_same_v<produce::Null,RemainingContainerT> ) {
                 setData<typename RemainingContainerT::head_type,
                       typename RemainingContainerT::tail_type>(iProducts);
               }
            }
         void newRecordComing() {
            wasCalledForThisRecord_ = false;
         }
         
     private:
         Callback(const Callback&) = delete; // stop default
         
         const Callback& operator=(const Callback&) = delete; // stop default

        std::array<void*, produce::size< TReturn >::value> proxyData_;
         edm::propagate_const<T*> producer_;
         method_type method_;
         bool wasCalledForThisRecord_;
         TDecorator decorator_;
      };
   }
}

#endif
