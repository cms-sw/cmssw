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
//

// system include files
#include <cassert>
#include <memory>

// user include files
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"

#include "FWCore/Framework/interface/produce_helpers.h"
#include "FWCore/Utilities/interface/propagate_const.h"

// forward declarations
namespace edm::eventsetup {

  template <class CallbackT, class RecordT, class DataT>
  class CallbackProxy : public DataProxy {
  public:
    using smart_pointer_traits = produce::smart_pointer_traits<DataT>;
    using ValueType = typename smart_pointer_traits::type;
    using RecordType = RecordT;

    CallbackProxy(std::shared_ptr<CallbackT>& iCallback) : callback_{iCallback} {
      //The callback fills the data directly.  This is done so that the callback does not have to
      //  hold onto a temporary copy of the result of the callback since the callback is allowed
      //  to return multiple items where only one item is needed by this Proxy
      iCallback->holdOntoPointer(&data_);
    }

    ~CallbackProxy() override {
      DataT* dummy(nullptr);
      callback_->holdOntoPointer(dummy);
    }

    const void* getImpl(const EventSetupRecordImpl& iRecord,
                        const DataKey&,
                        EventSetupImpl const* iEventSetupImpl) override {
      assert(iRecord.key() == RecordT::keyForClass());
      RecordType rec;
      rec.setImpl(&iRecord, callback_->transitionID(), callback_->getTokenIndices(), iEventSetupImpl, true);
      (*callback_)(rec);
      return smart_pointer_traits::getPointer(data_);
    }

    void invalidateCache() override {
      data_ = DataT{};
      callback_->newRecordComing();
    }

    // Delete copy operations
    CallbackProxy(const CallbackProxy&) = delete;
    const CallbackProxy& operator=(const CallbackProxy&) = delete;

  private:
    DataT data_{};
    edm::propagate_const<std::shared_ptr<CallbackT>> callback_;
  };

}  // namespace edm::eventsetup

#endif
