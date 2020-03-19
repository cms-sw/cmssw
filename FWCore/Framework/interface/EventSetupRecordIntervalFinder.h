#ifndef Framework_EventSetupRecordIntervalFinder_h
#define Framework_EventSetupRecordIntervalFinder_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordIntervalFinder
//
/**\class EventSetupRecordIntervalFinder EventSetupRecordIntervalFinder.h FWCore/Framework/interface/EventSetupRecordIntervalFinder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Tue Mar 29 16:15:11 EST 2005
//

// system include files
#include <map>
#include <set>

// user include files
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/ComponentDescription.h"

// forward declarations
namespace edm {

  class EventSetupRecordIntervalFinder {
  public:
    EventSetupRecordIntervalFinder() : intervals_() {}
    EventSetupRecordIntervalFinder(const EventSetupRecordIntervalFinder&) = delete;
    const EventSetupRecordIntervalFinder& operator=(const EventSetupRecordIntervalFinder&) = delete;
    virtual ~EventSetupRecordIntervalFinder() noexcept(false);

    // ---------- const member functions ---------------------
    std::set<eventsetup::EventSetupRecordKey> findingForRecords() const;

    const eventsetup::ComponentDescription& descriptionForFinder() const { return description_; }
    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
    /**returns the 'default constructed' ValidityInterval if no valid interval.
       If upperbound is not known, it should be set to IOVSyncValue::invalidIOVSyncValue()
      */
    const ValidityInterval& findIntervalFor(const eventsetup::EventSetupRecordKey&, const IOVSyncValue&);

    void resetInterval(const eventsetup::EventSetupRecordKey&);

    bool concurrentFinder() const { return isConcurrentFinder(); }

    bool nonconcurrentAndIOVNeedsUpdate(const eventsetup::EventSetupRecordKey& key,
                                        const IOVSyncValue& syncValue) const {
      return isNonconcurrentAndIOVNeedsUpdate(key, syncValue);
    }

    void setDescriptionForFinder(const eventsetup::ComponentDescription& iDescription) { description_ = iDescription; }

  protected:
    virtual void setIntervalFor(const eventsetup::EventSetupRecordKey&, const IOVSyncValue&, ValidityInterval&) = 0;

    template <class T>
    void findingRecord() {
      findingRecordWithKey(eventsetup::EventSetupRecordKey::makeKey<T>());
    }

    void findingRecordWithKey(const eventsetup::EventSetupRecordKey&);

  private:
    virtual void doResetInterval(const eventsetup::EventSetupRecordKey&);

    // The following function should be overridden in a derived class if
    // it supports concurrent IOVs. If this returns false (the default),
    // this will cause the Framework to run everything associated with the
    // particular IOV set in a call to the function setIntervalFor before
    // calling setIntervalFor again for that finder. In modules that do not
    // support concurrency, this time ordering is what the produce functions
    // use to know what IOV to produce data for. In finders that support
    // concurrency, the produce function will get information from the
    // Record to know what IOV to produce data for. This waiting to synchronize
    // IOV transitions can affect performance. It would be nice if someday
    // all finders were migrated to support concurrency and this function
    // and the code related to it could be deleted.
    virtual bool isConcurrentFinder() const;

    // Should only be overridden by DependentRecordIntervalFinder and
    // IntersectingIOVRecordIntervalFinder.
    virtual bool isNonconcurrentAndIOVNeedsUpdate(const eventsetup::EventSetupRecordKey&, const IOVSyncValue&) const;

    /** override this method if you need to delay setting what records you will be using until after all modules are loaded*/
    virtual void delaySettingRecords();
    // ---------- member data --------------------------------
    using Intervals = std::map<eventsetup::EventSetupRecordKey, ValidityInterval>;
    Intervals intervals_;

    eventsetup::ComponentDescription description_;
  };

}  // namespace edm
#endif
