// -*- C++ -*-
//
// Package:     Framework
// Module:      EventSetup
//
// Description: <one line class summary>
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu Mar 24 16:27:10 EST 2005
//

// system include files
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm {

  EventSetupImpl::EventSetupImpl() {}

  EventSetupImpl::~EventSetupImpl() {}

  void EventSetupImpl::insertRecordImpl(const eventsetup::EventSetupRecordKey& iKey,
                                        const eventsetup::EventSetupRecordImpl* iRecord) {
    auto lb = std::lower_bound(keysBegin_, keysEnd_, iKey);
    if (lb == keysEnd_ || iKey != *lb) {
      throw cms::Exception("LogicError") << "EventSetupImpl::insert Could not find key\n"
                                         << "Should be impossible. Please contact Framework developer.\n";
    }
    auto index = std::distance(keysBegin_, lb);
    recordImpls_[index] = iRecord;
  }

  void EventSetupImpl::addRecordImpl(const eventsetup::EventSetupRecordImpl& iRecord) {
    insertRecordImpl(iRecord.key(), &iRecord);
  }

  std::optional<eventsetup::EventSetupRecordGeneric> EventSetupImpl::find(const eventsetup::EventSetupRecordKey& iKey,
                                                                          unsigned int iTransitionID,
                                                                          ESProxyIndex const* getTokenIndices) const {
    auto lb = std::lower_bound(keysBegin_, keysEnd_, iKey);
    if (lb == keysEnd_ || iKey != *lb) {
      return std::nullopt;
    }
    auto index = std::distance(keysBegin_, lb);
    if (recordImpls_[index] == nullptr) {
      return std::nullopt;
    }
    return eventsetup::EventSetupRecordGeneric(recordImpls_[index], iTransitionID, getTokenIndices, this);
  }

  eventsetup::EventSetupRecordImpl const* EventSetupImpl::findImpl(const eventsetup::EventSetupRecordKey& iKey) const {
    auto lb = std::lower_bound(keysBegin_, keysEnd_, iKey);
    if (lb == keysEnd_ || iKey != *lb) {
      return nullptr;
    }
    auto index = std::distance(keysBegin_, lb);
    return recordImpls_[index];
  }

  eventsetup::EventSetupRecordImpl const* EventSetupImpl::findImpl(ESRecordIndex iKey) const {
    if UNLIKELY (iKey.value() == ESRecordIndex::invalidValue()) {
      return nullptr;
    }
    return recordImpls_[iKey.value()];
  }

  void EventSetupImpl::fillAvailableRecordKeys(std::vector<eventsetup::EventSetupRecordKey>& oToFill) const {
    oToFill.clear();
    oToFill.reserve(recordImpls_.size());

    for (auto const& recordImpl : recordImpls_) {
      if (recordImpl != nullptr) {
        oToFill.push_back(recordImpl->key());
      }
    }
  }

  bool EventSetupImpl::recordIsProvidedByAModule(eventsetup::EventSetupRecordKey const& iKey) const {
    auto lb = std::lower_bound(keysBegin_, keysEnd_, iKey);
    return lb != keysEnd_ && iKey == *lb;
  }

  bool EventSetupImpl::validRecord(eventsetup::EventSetupRecordKey const& iKey) const {
    auto lb = std::lower_bound(keysBegin_, keysEnd_, iKey);
    if (lb != keysEnd_ && iKey == *lb) {
      auto index = std::distance(keysBegin_, lb);
      return recordImpls_[index] != nullptr;
    }
    return false;
  }

  void EventSetupImpl::setKeyIters(std::vector<eventsetup::EventSetupRecordKey>::const_iterator const& keysBegin,
                                   std::vector<eventsetup::EventSetupRecordKey>::const_iterator const& keysEnd) {
    keysBegin_ = keysBegin;
    keysEnd_ = keysEnd;
    recordImpls_.resize(keysEnd_ - keysBegin_, nullptr);
  }

}  // namespace edm
