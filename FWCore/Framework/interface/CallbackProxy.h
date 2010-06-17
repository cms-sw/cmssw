#ifndef Framework_CallbackProxy_h
#define Framework_CallbackProxy_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     CallbackProxy
// 
/**\class CallbackProxy CallbackProxy.h FWCore/Framework/interface/CallbackProxy.h

 Description: A DataProxy which performs a callback when data is requested

 Usage:
    This class is primarily used by ESProducer to allow the EventSetup system
 to call a particular method of ESProducer where data is being requested.

*/
//
// Author:      Chris Jones
// Created:     Fri Apr  8 11:50:21 CDT 2005
// $Id: CallbackProxy.h,v 1.7 2006/08/26 18:38:32 chrjones Exp $
//

// system include files
#include "boost/shared_ptr.hpp"
#include <cassert>

// user include files
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"

#include "FWCore/Framework/interface/produce_helpers.h"

// forward declarations
namespace edm {
   namespace eventsetup {

      template<class CallbackT, class RecordT, class DataT>
      class CallbackProxy : public DataProxy {
         
      public:
         typedef  typename produce::smart_pointer_traits<DataT>::type value_type;
         typedef  RecordT record_type;
         
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
         const void* getImpl(const EventSetupRecord& iRecord, const DataKey&) {
            assert(iRecord.key() == RecordT::keyForClass());
            (*callback_)(static_cast<const record_type&>(iRecord));
            return &(*data_);
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
