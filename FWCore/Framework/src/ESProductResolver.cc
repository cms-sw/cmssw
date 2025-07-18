// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESProductResolver
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu Mar 31 12:49:19 EST 2005
//

#include "FWCore/Framework/interface/ESProductResolver.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/MakeDataException.h"
#include "FWCore/Framework/interface/EventSetupRecordImpl.h"
#include "FWCore/Utilities/interface/Likely.h"

namespace {
  constexpr int kInvalidLocation = 0;
  void const* const kInvalid = &kInvalidLocation;
}  // namespace

namespace edm {
  namespace eventsetup {

    static const ComponentDescription* dummyDescription() {
      static const ComponentDescription s_desc;
      return &s_desc;
    }

    ESProductResolver::ESProductResolver()
        : description_(dummyDescription()), cache_(kInvalid), nonTransientAccessRequested_(false) {}

    ESProductResolver::~ESProductResolver() {}

    bool ESProductResolver::cacheIsValid() const { return cache_.load() != kInvalid; }

    void ESProductResolver::clearCacheIsValid() {
      nonTransientAccessRequested_.store(false, std::memory_order_release);
      cache_.store(kInvalid);
    }

    void ESProductResolver::resetIfTransient() {
      if (!nonTransientAccessRequested_.load(std::memory_order_acquire)) {
        clearCacheIsValid();
        invalidateTransientCache();
      }
    }

    unsigned int ESProductResolver::produceMethodID() const { return 0; }

    void ESProductResolver::invalidateTransientCache() { invalidateCache(); }

    namespace {
      void throwMakeException(const EventSetupRecordImpl& iRecord, const DataKey& iKey) {
        throw MakeDataException(iRecord.key(), iKey);
      }

    }  // namespace

    void ESProductResolver::prefetchAsync(WaitingTaskHolder iTask,
                                          EventSetupRecordImpl const& iRecord,
                                          DataKey const& iKey,
                                          EventSetupImpl const* iEventSetupImpl,
                                          ServiceToken const& iToken,
                                          ESParentContext const& iParent) const noexcept {
      const_cast<ESProductResolver*>(this)->prefetchAsyncImpl(iTask, iRecord, iKey, iEventSetupImpl, iToken, iParent);
    }

    void const* ESProductResolver::getAfterPrefetch(const EventSetupRecordImpl& iRecord,
                                                    const DataKey& iKey,
                                                    bool iTransiently) const {
      //We need to set the AccessType for each request so this can't be called in an earlier function in the stack.
      //This also must be before the cache_ check since we want to setCacheIsValid before a possible
      // exception throw. If we don't, 'getImpl' will be called again on a second request for the data.

      if LIKELY (!iTransiently) {
        nonTransientAccessRequested_.store(true, std::memory_order_release);
      }

      auto cache = cache_.load();
      if UNLIKELY (cache == kInvalid) {
        // This is safe even if multiple threads get in here simultaneously
        // because cache_ is atomic and getAfterPrefetchImpl will return
        // the same pointer on all threads for the same IOV.
        // This is fast because the vast majority of the time only 1 thread per IOV
        // will get in here so most of the time only 1 atomic operation.
        cache = cache_ = getAfterPrefetchImpl();
      }
      if UNLIKELY (cache == nullptr) {
        throwMakeException(iRecord, iKey);
      }
      return cache;
    }

  }  // namespace eventsetup
}  // namespace edm
