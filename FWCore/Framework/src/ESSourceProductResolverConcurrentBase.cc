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

//
// member functions
//

void edm::eventsetup::ESSourceProductResolverConcurrentBase::prefetchAsyncImpl(
    edm::WaitingTaskHolder iTask,
    edm::eventsetup::EventSetupRecordImpl const& iRecord,
    edm::eventsetup::DataKey const& iKey,
    edm::EventSetupImpl const*,
    edm::ServiceToken const&,
    edm::ESParentContext const& iParent) {
  prefetchAsyncImplTemplate([](auto& iGroup, auto iActivity) { iGroup.run(std::move(iActivity)); },
                            []() { return true; },
                            std::move(iTask),
                            iRecord,
                            iKey,
                            iParent);
}
