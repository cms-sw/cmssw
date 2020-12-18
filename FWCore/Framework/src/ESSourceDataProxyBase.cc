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

//
// member functions
//

void edm::eventsetup::ESSourceDataProxyBase::prefetchAsyncImpl(edm::WaitingTaskHolder iTask,
                                                               edm::eventsetup::EventSetupRecordImpl const& iRecord,
                                                               edm::eventsetup::DataKey const& iKey,
                                                               edm::EventSetupImpl const*,
                                                               edm::ServiceToken const&) {
  bool expected = false;
  auto doPrefetch = m_prefetching.compare_exchange_strong(expected, true);
  m_waitingList.add(iTask);
  if (doPrefetch) {
    m_queue->push([this, iKey, &iRecord]() {
      try {
        {
          std::lock_guard<std::mutex> guard(*m_mutex);
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
