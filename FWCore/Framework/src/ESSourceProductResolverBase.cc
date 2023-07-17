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
#include <fmt/format.h>

// user include files
#include "FWCore/Framework/interface/ESSourceProductResolverBase.h"
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/ConvertException.h"

namespace edm {
  void exceptionContext(cms::Exception&, ESModuleCallingContext const&);
}

//
// member functions
//

bool edm::eventsetup::ESSourceProductResolverBase::needToPrefetch(edm::WaitingTaskHolder iTask) {
  m_waitingList.add(std::move(iTask));
  bool expected = false;
  return m_prefetching.compare_exchange_strong(expected, true);
}

void edm::eventsetup::ESSourceProductResolverBase::doPrefetchAndSignals(
    edm::eventsetup::EventSetupRecordImpl const& iRecord,
    edm::eventsetup::DataKey const& iKey,
    edm::ESParentContext const& iParent) {
  edm::ESModuleCallingContext context(providerDescription(), ESModuleCallingContext::State::kRunning, iParent);
  iRecord.activityRegistry()->preESModuleSignal_.emit(iRecord.key(), context);
  struct EndGuard {
    EndGuard(EventSetupRecordImpl const& iRecord, ESModuleCallingContext const& iContext)
        : record_{iRecord}, context_{iContext} {}
    ~EndGuard() { record_.activityRegistry()->postESModuleSignal_.emit(record_.key(), context_); }
    EventSetupRecordImpl const& record_;
    ESModuleCallingContext const& context_;
  } guardAR(iRecord, context);
  try {
    convertException::wrap([&] { prefetch(iKey, EventSetupRecordDetails(&iRecord)); });
  } catch (cms::Exception& iException) {
    iException.addContext(fmt::format(
        "Retrieving data product {} '{}' in record {}", iKey.type().name(), iKey.name().value(), iRecord.key().name()));
    exceptionContext(iException, context);
    throw;
  }
}

//
// const member functions
//

//
// static member functions
//
