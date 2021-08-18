// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     __class__
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  __author__
//         Created:  __date__
//

// system include files

// user include files
#include "FWCore/Framework/interface/ESSourceDataProxyBase.h"
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

//
// member functions
//

void edm::eventsetup::ESSourceDataProxyBase::prefetchAsyncImpl(edm::WaitingTaskHolder iTask,
                                                               edm::eventsetup::EventSetupRecordImpl const& iRecord,
                                                               edm::eventsetup::DataKey const& iKey,
                                                               edm::EventSetupImpl const*,
                                                               edm::ServiceToken const&,
                                                               edm::ESParentContext const& iParent) {
  bool expected = false;
  auto doPrefetch = m_prefetching.compare_exchange_strong(expected, true);
  m_waitingList.add(iTask);
  if (doPrefetch) {
    m_queue->push(*iTask.group(), [this, iKey, &iRecord, iParent]() {
      try {
        {
          std::lock_guard<std::mutex> guard(*m_mutex);
          edm::ESModuleCallingContext context(providerDescription(), ESModuleCallingContext::State::kRunning, iParent);
          iRecord.activityRegistry()->preESModuleSignal_.emit(iRecord.key(), context);
          struct EndGuard {
            EndGuard(EventSetupRecordImpl const& iRecord, ESModuleCallingContext const& iContext)
                : record_{iRecord}, context_{iContext} {}
            ~EndGuard() { record_.activityRegistry()->postESModuleSignal_.emit(record_.key(), context_); }
            EventSetupRecordImpl const& record_;
            ESModuleCallingContext const& context_;
          } guardAR(iRecord, context);
          prefetch(iKey, EventSetupRecordDetails(&iRecord));
        }
        m_waitingList.doneWaiting(std::exception_ptr{});
      } catch (...) {
        m_waitingList.doneWaiting(std::current_exception());
      }
    });
  }
}

//
// const member functions
//

//
// static member functions
//
