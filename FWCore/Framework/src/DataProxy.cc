// -*- C++ -*-
//
// Package:     Framework
// Class  :     DataProxy
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu Mar 31 12:49:19 EST 2005
//

#define TBB_PREVIEW_RESUMABLE_TASKS 1
// system include files
#include <mutex>

// user include files
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/MakeDataException.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

#include "FWCore/Framework/src/esTaskArenas.h"

#include "tbb/task.h"
#include "tbb/task_group.h"

namespace edm {
  namespace eventsetup {

    static const ComponentDescription* dummyDescription() {
      static ComponentDescription s_desc;
      return &s_desc;
    }

    DataProxy::DataProxy()
        : description_(dummyDescription()),
          cache_(nullptr),
          cacheIsValid_(false),
          nonTransientAccessRequested_(false) {}

    DataProxy::~DataProxy() {}

    void DataProxy::clearCacheIsValid() {
      nonTransientAccessRequested_.store(false, std::memory_order_release);
      cache_ = nullptr;
      cacheIsValid_.store(false, std::memory_order_release);
    }

    void DataProxy::resetIfTransient() {
      if (!nonTransientAccessRequested_.load(std::memory_order_acquire)) {
        clearCacheIsValid();
        invalidateTransientCache();
      }
    }

    void DataProxy::invalidateTransientCache() { invalidateCache(); }

    namespace {
      void throwMakeException(const EventSetupRecordImpl& iRecord, const DataKey& iKey) {
        throw MakeDataException(iRecord.key(), iKey);
      }

    }  // namespace

    void DataProxy::prefetchAsync(WaitingTaskHolder iTask,
                                  EventSetupRecordImpl const& iRecord,
                                  DataKey const& iKey,
                                  EventSetupImpl const* iEventSetupImpl,
                                  ServiceToken const& iToken,
                                  ESParentContext const& iParent) const {
      const_cast<DataProxy*>(this)->prefetchAsyncImpl(iTask, iRecord, iKey, iEventSetupImpl, iToken, iParent);
    }

    void const* DataProxy::getAfterPrefetch(const EventSetupRecordImpl& iRecord,
                                            const DataKey& iKey,
                                            bool iTransiently) const {
      //We need to set the AccessType for each request so this can't be called in an earlier function in the stack.
      //This also must be before the cache_ check since we want to setCacheIsValid before a possible
      // exception throw. If we don't, 'getImpl' will be called again on a second request for the data.

      if LIKELY (!iTransiently) {
        nonTransientAccessRequested_.store(true, std::memory_order_release);
      }

      if UNLIKELY (!cacheIsValid()) {
        cache_ = getAfterPrefetchImpl();
        cacheIsValid_.store(true, std::memory_order_release);
      }

      if UNLIKELY (nullptr == cache_) {
        throwMakeException(iRecord, iKey);
      }
      return cache_;
    }

    const void* DataProxy::get(const EventSetupRecordImpl& iRecord,
                               const DataKey& iKey,
                               bool iTransiently,
                               ActivityRegistry const* activityRegistry,
                               EventSetupImpl const* iEventSetupImpl,
                               ESParentContext const& iParent) const {
      if (!cacheIsValid()) {
        auto token = ServiceRegistry::instance().presentToken();
        std::exception_ptr exceptPtr{};
        edm::esTaskArena().execute([this, &exceptPtr, &iRecord, &iKey, iEventSetupImpl, token, iParent]() {
          //tbb::task::suspend can only be run from within a task running in this arena. For 1 thread,
          // it is often (always?) the case where not such task is being run here. Therefore we need
          // to use a temp task_group to start up such a task.
          tbb::task_group group;
          group.run([&]() {
            tbb::task::suspend([&, this](tbb::task::suspend_point tag) {
              auto waitTask = make_waiting_task([tag, &exceptPtr](std::exception_ptr const* iExcept) {
                if (iExcept) {
                  exceptPtr = *iExcept;
                }
                tbb::task::resume(tag);
              });
              prefetchAsync(WaitingTaskHolder(group, waitTask), iRecord, iKey, iEventSetupImpl, token, iParent);
            });  //suspend
          });    //group.run
          group.wait();
        });  //esTaskArena().execute
        cache_ = getAfterPrefetchImpl();
        cacheIsValid_.store(true, std::memory_order_release);
        if (exceptPtr) {
          std::rethrow_exception(exceptPtr);
        }
      }
      return getAfterPrefetch(iRecord, iKey, iTransiently);
    }

  }  // namespace eventsetup
}  // namespace edm
