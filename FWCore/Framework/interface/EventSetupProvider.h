// -*- C++ -*-
#ifndef FWCore_Framework_EventSetupProvider_h
#define FWCore_Framework_EventSetupProvider_h
//
// Package:     Framework
// Class:      EventSetupProvider
//
/**\class edm::eventsetup::EventSetupProvider

 Description: Factory for a EventSetup

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu Mar 24 14:10:07 EST 2005
//

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "FWCore/Utilities/interface/propagate_const.h"

// forward declarations
namespace edm {
  class ActivityRegistry;
  class EventSetupImpl;
  class EventSetupRecordIntervalFinder;
  class IOVSyncValue;

  namespace eventsetup {
    struct ComponentDescription;
    class DataKey;
    class ESProductResolverProvider;
    class ESRecordsToProductResolverIndices;
    class EventSetupRecordKey;
    class EventSetupRecordProvider;
    class NumberOfConcurrentIOVs;

    class EventSetupProvider {
    public:
      typedef std::string RecordName;
      typedef std::string DataType;
      typedef std::string DataLabel;
      typedef std::pair<DataType, DataLabel> DataKeyInfo;
      typedef std::multimap<RecordName, DataKeyInfo> RecordToDataMap;
      typedef std::map<ComponentDescription, RecordToDataMap> PreferredProviderInfo;

      EventSetupProvider(ActivityRegistry const*, PreferredProviderInfo const* iInfo = nullptr);
      EventSetupProvider(EventSetupProvider const&) = delete;
      EventSetupProvider const& operator=(EventSetupProvider const&) = delete;

      ~EventSetupProvider();

      std::set<ComponentDescription> resolverProviderDescriptions() const;

      ESRecordsToProductResolverIndices recordsToResolverIndices() const;

      ///Set the validity intervals in all EventSetupRecordProviders
      void setAllValidityIntervals(const IOVSyncValue& iValue);

      std::shared_ptr<const EventSetupImpl> eventSetupForInstance(IOVSyncValue const&, bool& newEventSetupImpl);

      bool doWeNeedToWaitForIOVsToFinish(IOVSyncValue const&) const;

      EventSetupImpl const& eventSetupImpl() const { return *eventSetupImpl_; }

      void add(std::shared_ptr<ESProductResolverProvider>);
      void add(std::shared_ptr<EventSetupRecordIntervalFinder>);

      void finishConfiguration(NumberOfConcurrentIOVs const&, bool& hasNonconcurrentFinder);

      ///Used when we need to force a Record to reset all its resolvers
      void resetRecordPlusDependentRecords(EventSetupRecordKey const&);

      ///Used when testing that all code properly updates on IOV changes of all Records
      void forceCacheClear();

      void updateLookup();

      void clearInitializationData();

      /// Intended for use only in tests
      void addRecord(const EventSetupRecordKey& iKey);
      void setPreferredProviderInfo(PreferredProviderInfo const& iInfo);

      void fillRecordsNotAllowingConcurrentIOVs(std::set<EventSetupRecordKey>& recordsNotAllowingConcurrentIOVs) const;

      void fillKeys(std::set<EventSetupRecordKey>& keys) const;

      EventSetupRecordProvider* tryToGetRecordProvider(const EventSetupRecordKey& iKey);
      EventSetupRecordProvider const* tryToGetRecordProvider(const EventSetupRecordKey& iKey) const;

      void fillAllESProductResolverProviders(std::vector<ESProductResolverProvider const*>&) const;

      ActivityRegistry const* activityRegistry() const { return activityRegistry_; }

    private:
      std::shared_ptr<EventSetupRecordProvider>& recordProvider(const EventSetupRecordKey& iKey);
      void insert(EventSetupRecordKey const&, std::unique_ptr<EventSetupRecordProvider>);

      void determinePreferred();

      // ---------- member data --------------------------------

      using RecordKeys = std::vector<EventSetupRecordKey>;
      RecordKeys recordKeys_;

      using RecordProviders = std::vector<std::shared_ptr<EventSetupRecordProvider>>;
      RecordProviders recordProviders_;

      ActivityRegistry const* activityRegistry_;

      propagate_const<std::shared_ptr<EventSetupImpl>> eventSetupImpl_;

      // The following are all used only during initialization and then cleared.

      std::unique_ptr<PreferredProviderInfo> preferredProviderInfo_;
      std::unique_ptr<std::vector<std::shared_ptr<EventSetupRecordIntervalFinder>>> finders_;
      std::unique_ptr<std::vector<std::shared_ptr<ESProductResolverProvider>>> dataProviders_;
      std::unique_ptr<std::map<EventSetupRecordKey, std::map<DataKey, ComponentDescription>>> recordToPreferred_;
    };

  }  // namespace eventsetup
}  // namespace edm
#endif
