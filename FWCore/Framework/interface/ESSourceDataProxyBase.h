#ifndef FWCore_Framework_ESSourceDataProxyBase_h
#define FWCore_Framework_ESSourceDataProxyBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ESSourceDataProxyBase
//
/**\class ESSourceDataProxyBase ESSourceDataProxyBase.h "FWCore/Framework/interface/ESSourceDataProxyBase.h"

 Description: Base class for DataProxies for ESSources that require synchronization

 Usage:
    The ESSourceDataProxyBase uses a SerialTaskQueue to serialize all DataProxies for the ESSource and a
    std::mutex to protect from concurrent calls to a DataProxy and the ESSource itself. Such concurrent calls
    can happen if concurrent LuminosityBlocks are being used.

    NOTE: if inheriting classes override `void invalidateCache()` they must be sure to call this classes
    implementation as part of the call.

*/
//
// Original Author:  Chris Jones
//         Created:  14/05/2020
//

// system include files
#include <mutex>
#include <atomic>

// user include files
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/EventSetupRecordDetails.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/Concurrency/interface/SerialTaskQueue.h"

// forward declarations

namespace edm::eventsetup {
  class ESSourceDataProxyBase : public DataProxy {
  public:
    ESSourceDataProxyBase(edm::SerialTaskQueue* iQueue, std::mutex* iMutex)
        : m_queue(iQueue), m_mutex(iMutex), m_prefetching{false} {}

    edm::SerialTaskQueue* queue() const { return m_queue; }
    std::mutex* mutex() const { return m_mutex; }

  protected:
    void invalidateCache() override {
      m_waitingList.reset();
      m_prefetching = false;
    }
    void invalidateTransientCache() override {}

    virtual void prefetch(edm::eventsetup::DataKey const& iKey, EventSetupRecordDetails) = 0;

  private:
    void prefetchAsyncImpl(edm::WaitingTaskHolder iTask,
                           edm::eventsetup::EventSetupRecordImpl const&,
                           edm::eventsetup::DataKey const& iKey,
                           edm::EventSetupImpl const*,
                           edm::ServiceToken const&) final;

    // ---------- member data --------------------------------

    edm::WaitingTaskList m_waitingList;
    edm::SerialTaskQueue* m_queue;
    std::mutex* m_mutex;
    std::atomic<bool> m_prefetching;
  };
}  // namespace edm::eventsetup
#endif
