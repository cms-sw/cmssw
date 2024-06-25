#ifndef FWCore_Framework_ESSourceProductResolverBase_h
#define FWCore_Framework_ESSourceProductResolverBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ESSourceProductResolverBase
//
/**\class ESSourceProductResolverBase ESSourceProductResolverBase.h "FWCore/Framework/interface/ESSourceProductResolverBase.h"

 Description: Base class for DataProxies for ESSources that can be specialized based on concurrency needs

 Usage:
    The ESSourceProductResolverBase provides the bases for DataProxies needed for ESSources. It allows customization of synchronization needs via the use of template parameters.

    NOTE: if inheriting classes override `void invalidateCache()` they must be sure to call this classes
    implementation as part of the call.

*/
//
// Original Author:  Chris Jones
//         Created:  14/05/2020
//

// system include files
#include <atomic>

// user include files
#include "FWCore/Framework/interface/ESProductResolver.h"
#include "FWCore/Framework/interface/EventSetupRecordDetails.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"

// forward declarations

namespace edm::eventsetup {
  class ESSourceProductResolverBase : public ESProductResolver {
  public:
    ESSourceProductResolverBase() : m_prefetching{false} {}

  protected:
    void invalidateCache() override {
      m_waitingList.reset();
      m_prefetching = false;
    }
    void invalidateTransientCache() override {}

    virtual void prefetch(edm::eventsetup::DataKey const& iKey, EventSetupRecordDetails) = 0;

    //Should call from prefetchAsyncImpl
    template <typename ASYNC, typename GUARD>
    void prefetchAsyncImplTemplate(ASYNC iAsync,
                                   GUARD iGuardFactory,
                                   edm::WaitingTaskHolder iTask,
                                   edm::eventsetup::EventSetupRecordImpl const& iRecord,
                                   edm::eventsetup::DataKey const& iKey,
                                   edm::ESParentContext const& iContext) noexcept {
      auto group = iTask.group();
      if (needToPrefetch(std::move(iTask))) {
        iAsync(*group, [this, iGuardFactory, &iRecord, iKey, iContext]() {
          try {
            guardPrefetch(iGuardFactory, iRecord, iKey, iContext);
            m_waitingList.doneWaiting(std::exception_ptr{});
          } catch (...) {
            m_waitingList.doneWaiting(std::current_exception());
          }
        });
      }
    }

  private:
    template <typename GUARD>
    void guardPrefetch(GUARD iGuardFactory,
                       edm::eventsetup::EventSetupRecordImpl const& iES,
                       edm::eventsetup::DataKey const& iKey,
                       edm::ESParentContext const& iContext) {
      [[maybe_unused]] auto guard = iGuardFactory();
      doPrefetchAndSignals(iES, iKey, iContext);
    }

    bool needToPrefetch(edm::WaitingTaskHolder iTask) noexcept;

    void doPrefetchAndSignals(edm::eventsetup::EventSetupRecordImpl const&,
                              edm::eventsetup::DataKey const& iKey,
                              edm::ESParentContext const&);

    // ---------- member data --------------------------------

    edm::WaitingTaskList m_waitingList;
    std::atomic<bool> m_prefetching;
  };
}  // namespace edm::eventsetup
#endif
