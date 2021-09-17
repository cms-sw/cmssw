// -*- C++ -*-
//
// Package:     Framework
// Class  :     DependentRecordIntervalFinder
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Sat Apr 30 19:37:22 EDT 2005
//

#include "FWCore/Framework/interface/DependentRecordIntervalFinder.h"
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <cassert>

namespace edm {
  namespace eventsetup {

    DependentRecordIntervalFinder::DependentRecordIntervalFinder(const EventSetupRecordKey& iKey) : providers_() {
      findingRecordWithKey(iKey);
    }

    DependentRecordIntervalFinder::~DependentRecordIntervalFinder() {}

    void DependentRecordIntervalFinder::addProviderWeAreDependentOn(
        std::shared_ptr<EventSetupRecordProvider> iProvider) {
      providers_.push_back(iProvider);
    }

    void DependentRecordIntervalFinder::setAlternateFinder(std::shared_ptr<EventSetupRecordIntervalFinder> iOther) {
      alternate_ = iOther;
    }

    void DependentRecordIntervalFinder::setIntervalFor(const EventSetupRecordKey& iKey,
                                                       const IOVSyncValue& iTime,
                                                       ValidityInterval& oInterval) {
      //NOTE: oInterval is the last value that was used so if nothing changed do not modify oInterval

      //I am assuming that an invalidTime is always less then the first valid time
      assert(IOVSyncValue::invalidIOVSyncValue() < IOVSyncValue::beginOfTime());
      if (providers_.empty() && alternate_.get() == nullptr) {
        oInterval = ValidityInterval::invalidInterval();
        return;
      }
      bool haveAValidDependentRecord = false;
      bool allRecordsValid = true;
      ValidityInterval newInterval(IOVSyncValue::beginOfTime(), IOVSyncValue::endOfTime());

      if (alternate_.get() != nullptr) {
        ValidityInterval test = alternate_->findIntervalFor(iKey, iTime);
        if (test != ValidityInterval::invalidInterval()) {
          haveAValidDependentRecord = true;
          newInterval = test;
        }
      }
      bool intervalsWereComparible = true;
      for (Providers::iterator itProvider = providers_.begin(), itProviderEnd = providers_.end();
           itProvider != itProviderEnd;
           ++itProvider) {
        if ((*itProvider)->setValidityIntervalFor(iTime)) {
          haveAValidDependentRecord = true;
          ValidityInterval providerInterval = (*itProvider)->validityInterval();
          if ((!newInterval.first().comparable(providerInterval.first())) ||
              (!newInterval.last().comparable(providerInterval.last()))) {
            intervalsWereComparible = false;
            break;
          }
          if (newInterval.first() < providerInterval.first()) {
            newInterval.setFirst(providerInterval.first());
          }
          if (newInterval.last() > providerInterval.last()) {
            newInterval.setLast(providerInterval.last());
          }
        } else {
          allRecordsValid = false;
        }
      }
      if (intervalsWereComparible) {
        if (!haveAValidDependentRecord) {
          //If no Finder has no valid time, then this record is also invalid for this time
          newInterval = ValidityInterval::invalidInterval();
        }
        if (!allRecordsValid) {
          //since some of the dependent providers do not have a valid IOV for this syncvalue
          // we do not know what the true end IOV is.  Therefore we must make it open ended
          // so that we can check each time to see if those providers become valid.
          newInterval.setLast(IOVSyncValue::invalidIOVSyncValue());
        }
        oInterval = newInterval;
        return;
      }
      //handle the case where some providers use time and others use run/lumi/event
      // in this case all we can do is find an IOV which changed since last time
      // and use its start time to do the synching and use an 'invalid' end time
      // so the system always checks back to see if something has changed
      if (previousIOVs_.empty()) {
        std::vector<ValidityInterval> tmp(providers_.size(), ValidityInterval());
        previousIOVs_.swap(tmp);
      }

      //I'm using an heuristic to pick a reasonable starting point for the IOV. The idea is to
      // assume that lumi sections are 23 seconds long and therefore if we take a difference between
      // iTime and the beginning of a changed IOV we can pick the changed IOV with the start time
      // closest to iTime. This doesn't have to be perfect, we just want to reduce the dependency
      // on provider order to make the jobs more deterministic

      bool hadChangedIOV = false;
      //both start at the smallest value
      EventID closestID;
      Timestamp closestTimeStamp(0);
      std::vector<ValidityInterval>::iterator itIOVs = previousIOVs_.begin();
      for (Providers::iterator itProvider = providers_.begin(), itProviderEnd = providers_.end();
           itProvider != itProviderEnd;
           ++itProvider, ++itIOVs) {
        if ((*itProvider)->setValidityIntervalFor(iTime)) {
          ValidityInterval providerInterval = (*itProvider)->validityInterval();
          if (*itIOVs != providerInterval) {
            hadChangedIOV = true;
            if (providerInterval.first().time().value() == 0) {
              //this is a run/lumi based one
              if (closestID < providerInterval.first().eventID()) {
                closestID = providerInterval.first().eventID();
              }
            } else {
              if (closestTimeStamp < providerInterval.first().time()) {
                closestTimeStamp = providerInterval.first().time();
              }
            }
            *itIOVs = providerInterval;
          }
        }
      }
      if (hadChangedIOV) {
        if (closestID.run() != 0) {
          if (closestTimeStamp.value() == 0) {
            //no time
            oInterval = ValidityInterval(IOVSyncValue(closestID), IOVSyncValue::invalidIOVSyncValue());
          } else {
            if (closestID.run() == iTime.eventID().run()) {
              //can compare time to lumi
              const unsigned long long kLumiTimeLength = 23;

              if ((iTime.eventID().luminosityBlock() - closestID.luminosityBlock()) * kLumiTimeLength <
                  iTime.time().unixTime() - closestTimeStamp.unixTime()) {
                //closestID was closer
                oInterval = ValidityInterval(IOVSyncValue(closestID), IOVSyncValue::invalidIOVSyncValue());
              } else {
                oInterval = ValidityInterval(IOVSyncValue(closestTimeStamp), IOVSyncValue::invalidIOVSyncValue());
              }
            } else {
              //since we don't know how to change run # into time we can't compare
              // so if we have a time just use it
              oInterval = ValidityInterval(IOVSyncValue(closestTimeStamp), IOVSyncValue::invalidIOVSyncValue());
            }
          }
        } else {
          oInterval = ValidityInterval(IOVSyncValue(closestTimeStamp), IOVSyncValue::invalidIOVSyncValue());
        }
      }
    }

    void DependentRecordIntervalFinder::doResetInterval(const eventsetup::EventSetupRecordKey& key) {
      if (alternate_.get()) {
        alternate_->resetInterval(key);
      }
      previousIOVs_.clear();
    }

    bool DependentRecordIntervalFinder::isConcurrentFinder() const {
      throw Exception(errors::LogicError)
          << "DependentRecordIntervalFinder::isConcurrentFinder() should never be called.\n"
          << "Contact a Framework developer\n";
      return true;
    }

    bool DependentRecordIntervalFinder::isNonconcurrentAndIOVNeedsUpdate(const EventSetupRecordKey& iKey,
                                                                         const IOVSyncValue& iTime) const {
      // Note that we do not worry about dependent records here because this function
      // will get called once for every record and we would just be checking the
      // dependent records multiple times if we checked them inside this function.
      if (alternate_.get()) {
        return alternate_->nonconcurrentAndIOVNeedsUpdate(iKey, iTime);
      }
      return false;
    }

  }  // namespace eventsetup
}  // namespace edm
