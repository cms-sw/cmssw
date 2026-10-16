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
#include "FWCore/Framework/interface/ESSourceProductResolverConcurrentBase.h"
#include "FWCore/Concurrency/interface/StreamNumaDomains.h"

//
// member functions
//

void edm::eventsetup::ESSourceProductResolverConcurrentBase::prefetchAsyncImpl(
    edm::WaitingTaskHolder iTask,
    edm::eventsetup::EventSetupRecordImpl const& iRecord,
    edm::eventsetup::DataKey const& iKey,
    edm::EventSetupImpl const*,
    edm::ServiceToken const&,
    edm::ESParentContext const& iParent) noexcept {
  // with EventSetup replication, produce the copy of a NUMA domain in the arena of that domain
  prefetchAsyncImplTemplate([domain = iRecord.numaDomain()](auto& iGroup, auto iActivity) {
                              edm::numa::runInDomain(iGroup, domain, std::move(iActivity));
                            },
                            []() { return true; },
                            std::move(iTask),
                            iRecord,
                            iKey,
                            iParent);
}
