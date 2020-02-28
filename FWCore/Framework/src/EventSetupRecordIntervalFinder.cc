// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordIntervalFinder
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Wed Mar 30 14:27:26 EST 2005
//

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/src/ESGlobalMutex.h"
#include "FWCore/Utilities/interface/Likely.h"
#include <cassert>
#include <mutex>

using namespace edm::eventsetup;
namespace edm {

  EventSetupRecordIntervalFinder::~EventSetupRecordIntervalFinder() noexcept(false) {}

  const ValidityInterval& EventSetupRecordIntervalFinder::findIntervalFor(const EventSetupRecordKey& iKey,
                                                                          const IOVSyncValue& iInstance) {
    Intervals::iterator itFound = intervals_.find(iKey);
    assert(itFound != intervals_.end());
    if (!itFound->second.validFor(iInstance)) {
      if
        LIKELY(iInstance != IOVSyncValue::invalidIOVSyncValue()) {
          std::lock_guard<std::recursive_mutex> guard(esGlobalMutex());
          setIntervalFor(iKey, iInstance, itFound->second);
        }
      else {
        itFound->second = ValidityInterval::invalidInterval();
      }
    }
    return itFound->second;
  }

  void EventSetupRecordIntervalFinder::resetInterval(const eventsetup::EventSetupRecordKey& iKey) {
    Intervals::iterator itFound = intervals_.find(iKey);
    assert(itFound != intervals_.end());
    itFound->second = ValidityInterval{};
    doResetInterval(iKey);
  }

  void EventSetupRecordIntervalFinder::findingRecordWithKey(const EventSetupRecordKey& iKey) {
    intervals_.insert(Intervals::value_type(iKey, ValidityInterval()));
  }

  void EventSetupRecordIntervalFinder::doResetInterval(const eventsetup::EventSetupRecordKey&) {}

  bool EventSetupRecordIntervalFinder::isConcurrentFinder() const { return false; }

  bool EventSetupRecordIntervalFinder::isNonconcurrentAndIOVNeedsUpdate(const eventsetup::EventSetupRecordKey& iKey,
                                                                        const IOVSyncValue& iTime) const {
    if (!isConcurrentFinder()) {
      if (iTime == IOVSyncValue::invalidIOVSyncValue()) {
        return true;
      }
      Intervals::const_iterator itFound = intervals_.find(iKey);
      assert(itFound != intervals_.end());
      return !itFound->second.validFor(iTime);
    }
    return false;
  }

  void EventSetupRecordIntervalFinder::delaySettingRecords() {}

  std::set<EventSetupRecordKey> EventSetupRecordIntervalFinder::findingForRecords() const {
    if (intervals_.empty()) {
      //we are delaying our reading
      const_cast<EventSetupRecordIntervalFinder*>(this)->delaySettingRecords();
    }

    std::set<EventSetupRecordKey> returnValue;

    for (Intervals::const_iterator itEntry = intervals_.begin(), itEntryEnd = intervals_.end(); itEntry != itEntryEnd;
         ++itEntry) {
      returnValue.insert(returnValue.end(), itEntry->first);
    }
    return returnValue;
  }
}  // namespace edm
