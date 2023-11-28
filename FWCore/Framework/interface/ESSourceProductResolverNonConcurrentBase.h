#ifndef FWCore_Framework_ESSourceProductResolverNonConcurrentBase_h
#define FWCore_Framework_ESSourceProductResolverNonConcurrentBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ESSourceProductResolverNonConcurrentBase
//
/**\class ESSourceProductResolverNonConcurrentBase ESSourceProductResolverNonConcurrentBase.h "FWCore/Framework/interface/ESSourceProductResolverNonConcurrentBase.h"

 Description: Base class for DataProxies for ESSources that require synchronization

 Usage:
    The ESSourceProductResolverNonConcurrentBase uses a SerialTaskQueue to serialize all DataProxies for the ESSource and a
    std::mutex to protect from concurrent calls to a ESProductResolver and the ESSource itself. Such concurrent calls
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
#include "FWCore/Framework/interface/ESSourceProductResolverBase.h"
#include "FWCore/Concurrency/interface/SerialTaskQueue.h"

// forward declarations

namespace edm::eventsetup {
  class ESSourceProductResolverNonConcurrentBase : public ESSourceProductResolverBase {
  public:
    ESSourceProductResolverNonConcurrentBase(edm::SerialTaskQueue* iQueue, std::mutex* iMutex)
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
