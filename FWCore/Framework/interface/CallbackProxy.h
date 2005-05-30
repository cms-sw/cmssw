#ifndef EVENTSETUPPRODUCER_CALLBACKPROXY_H
#define EVENTSETUPPRODUCER_CALLBACKPROXY_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     CallbackProxy
// 
/**\class CallbackProxy CallbackProxy.h Core/CoreFramework/interface/CallbackProxy.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Fri Apr  8 11:50:21 CDT 2005
// $Id: CallbackProxy.h,v 1.1 2005/05/29 02:29:53 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/CoreFramework/interface/DataProxyTemplate.h"

#include "FWCore/CoreFramework/interface/produce_helpers.h"

// forward declarations
namespace edm {
   namespace eventsetup {

template<class CallbackT, class RecordT, class DataT>
      class CallbackProxy : public DataProxyTemplate<RecordT, typename produce::smart_pointer_traits<DataT>::type >
{

   public:
   typedef typename DataProxyTemplate<RecordT, typename produce::smart_pointer_traits<DataT>::type >::value_type value_type;
   typedef typename DataProxyTemplate<RecordT, typename produce::smart_pointer_traits<DataT>::type >::record_type record_type;
   
   CallbackProxy(boost::shared_ptr<CallbackT>& iCallback) :
   callback_(iCallback) { 
      //The callback fills the data directly.  This is done so that the callback does not have to
      //  hold onto a temporary copy of the result of the callback since the callback is allowed
      //  to return multiple items where only one item is needed by this Proxy
      iCallback->holdOntoPointer( &data_ ) ; }
      virtual ~CallbackProxy() {
         DataT* dummy(0);
         callback_->holdOntoPointer( dummy ) ;
      }
      // ---------- const member functions ---------------------
   
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      const value_type* make( const record_type& iRecord, const DataKey&) {
         (*callback_)( iRecord);
         return &(*data_) ;
      }
      void invalidateCache() {
         data_ = DataT();
         callback_->newRecordComing();
      }
   private:
      CallbackProxy( const CallbackProxy& ); // stop default

      const CallbackProxy& operator=( const CallbackProxy& ); // stop default

      // ---------- member data --------------------------------
      DataT data_;
      boost::shared_ptr<CallbackT> callback_;
};

   }
}

#endif /* EVENTSETUPPRODUCER_CALLBACKPROXY_H */
