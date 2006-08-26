#ifndef Framework_CallbackProxy_h
#define Framework_CallbackProxy_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     CallbackProxy
// 
/**\class CallbackProxy CallbackProxy.h FWCore/Framework/interface/CallbackProxy.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Fri Apr  8 11:50:21 CDT 2005
// $Id: CallbackProxy.h,v 1.6 2005/09/01 05:44:39 wmtan Exp $
//

// system include files
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/DataProxyTemplate.h"

#include "FWCore/Framework/interface/produce_helpers.h"

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
   data_(),
   callback_(iCallback) { 
      //The callback fills the data directly.  This is done so that the callback does not have to
      //  hold onto a temporary copy of the result of the callback since the callback is allowed
      //  to return multiple items where only one item is needed by this Proxy
      iCallback->holdOntoPointer(&data_) ; }
      virtual ~CallbackProxy() {
         DataT* dummy(0);
         callback_->holdOntoPointer(dummy) ;
      }
      // ---------- const member functions ---------------------
   
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      const value_type* make(const record_type& iRecord, const DataKey&) {
         (*callback_)(iRecord);
         return &(*data_) ;
      }
      void invalidateCache() {
         data_ = DataT();
         callback_->newRecordComing();
      }
   private:
      CallbackProxy(const CallbackProxy&); // stop default

      const CallbackProxy& operator=(const CallbackProxy&); // stop default

      // ---------- member data --------------------------------
      DataT data_;
      boost::shared_ptr<CallbackT> callback_;
};

   }
}

#endif
