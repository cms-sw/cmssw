#ifndef FWCore_Framework_ESSourceDataProxyNonConcurrentBase_h
#define FWCore_Framework_ESSourceDataProxyNonConcurrentBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ESSourceDataProxyNonConcurrentBase
//
/**\class ESSourceDataProxyNonConcurrentBase ESSourceDataProxyNonConcurrentBase.h "FWCore/Framework/interface/ESSourceDataProxyNonConcurrentBase.h"

 Description: Base class for DataProxies for ESSources that require synchronization

 Usage:
    The ESSourceDataProxyNonConcurrentBase uses a SerialTaskQueue to serialize all DataProxies for the ESSource and a
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

// user include files
#include "FWCore/Framework/interface/ESSourceDataProxyBase.h"
#include "FWCore/Concurrency/interface/SerialTaskQueue.h"

// forward declarations

namespace edm::eventsetup {
  class ESSourceDataProxyNonConcurrentBase : public ESSourceDataProxyBase {
  public:
    ESSourceDataProxyNonConcurrentBase(edm::SerialTaskQueue* iQueue, std::mutex* iMutex)
        : m_queue(iQueue), m_mutex(iMutex) {}

    edm::SerialTaskQueue* queue() const { return m_queue; }
    std::mutex* mutex() const { return m_mutex; }

  private:
    void prefetchAsyncImpl(edm::WaitingTaskHolder iTask,
                           edm::eventsetup::EventSetupRecordImpl const& iES,
                           edm::eventsetup::DataKey const& iKey,
                           edm::EventSetupImpl const*,
                           edm::ServiceToken const&,
                           edm::ESParentContext const&) final;

    // ---------- member data --------------------------------

    edm::SerialTaskQueue* m_queue;
    std::mutex* m_mutex;
  };
}  // namespace edm::eventsetup
#endif
