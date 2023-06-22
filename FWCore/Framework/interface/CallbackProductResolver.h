#ifndef Framework_CallbackProductResolver_h
#define Framework_CallbackProductResolver_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     CallbackProductResolver
//
/**\class CallbackProductResolver CallbackProductResolver.h FWCore/Framework/interface/CallbackProductResolver.h

 Description: A ESProductResolver which performs a callback when data is requested

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
#include "FWCore/Framework/interface/ESProductResolver.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

#include "FWCore/Framework/interface/produce_helpers.h"
#include "FWCore/Utilities/interface/propagate_const.h"

// forward declarations
namespace edm::eventsetup {

  template <class CallbackT, class RecordT, class DataT>
  class CallbackProductResolver final : public ESProductResolver {
  public:
    using smart_pointer_traits = produce::smart_pointer_traits<DataT>;
    using ValueType = typename smart_pointer_traits::type;
    using RecordType = RecordT;

    CallbackProductResolver(std::shared_ptr<CallbackT>& iCallback) : callback_{iCallback} {
      //The callback fills the data directly.  This is done so that the callback does not have to
      //  hold onto a temporary copy of the result of the callback since the callback is allowed
      //  to return multiple items where only one item is needed by this ProductResolver
      iCallback->holdOntoPointer(&data_);
    }

    ~CallbackProductResolver() final {
      DataT* dummy(nullptr);
      callback_->holdOntoPointer(dummy);
    }

    void prefetchAsyncImpl(WaitingTaskHolder iWaitTask,
                           const EventSetupRecordImpl& iRecord,
                           const DataKey&,
                           EventSetupImpl const* iEventSetupImpl,
                           ServiceToken const& iToken,
                           edm::ESParentContext const& iParent) final {
      assert(iRecord.key() == RecordT::keyForClass());
      callback_->prefetchAsync(iWaitTask, &iRecord, iEventSetupImpl, iToken, iParent);
    }

    void const* getAfterPrefetchImpl() const final { return smart_pointer_traits::getPointer(data_); }

    void invalidateCache() override {
      data_ = DataT{};
      callback_->newRecordComing();
    }

    // Delete copy operations
    CallbackProductResolver(const CallbackProductResolver&) = delete;
    const CallbackProductResolver& operator=(const CallbackProductResolver&) = delete;

  private:
    DataT data_{};
    edm::propagate_const<std::shared_ptr<CallbackT>> callback_;
  };

}  // namespace edm::eventsetup

#endif
