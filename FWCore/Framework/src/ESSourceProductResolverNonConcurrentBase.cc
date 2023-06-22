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
#include "FWCore/Framework/interface/ESSourceProductResolverNonConcurrentBase.h"

//
// member functions
//

void edm::eventsetup::ESSourceProductResolverNonConcurrentBase::prefetchAsyncImpl(
    edm::WaitingTaskHolder iTask,
    edm::eventsetup::EventSetupRecordImpl const& iRecord,
    edm::eventsetup::DataKey const& iKey,
    edm::EventSetupImpl const*,
    edm::ServiceToken const&,
    edm::ESParentContext const& iParent) {
  prefetchAsyncImplTemplate([this](auto& iGroup, auto iActivity) { m_queue->push(iGroup, std::move(iActivity)); },
                            [mutex = m_mutex]() { return std::lock_guard<std::mutex>(*mutex); },
                            std::move(iTask),
                            iRecord,
                            iKey,
                            iParent);
}
