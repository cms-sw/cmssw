#ifndef FWCore_Framework_EventSetupProvider_h
#define FWCore_Framework_EventSetupProvider_h
// -*- C++ -*-
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

#include "FWCore/Utilities/interface/propagate_const.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

// forward declarations
namespace edm {
  class ActivityRegistry;
  class EventSetupImpl;
  class EventSetupRecordIntervalFinder;
  class IOVSyncValue;
  class ModuleTypeResolverMaker;
  class ParameterSet;

  namespace eventsetup {
    struct ComponentDescription;
    class DataKey;
    class ESProductResolverProvider;
    class EventSetupRecordImpl;
    class EventSetupRecordKey;
    class EventSetupRecordProvider;
    class EventSetupsController;
    class NumberOfConcurrentIOVs;
    class ParameterSetIDHolder;
    class ESRecordsToProductResolverIndices;

    class EventSetupProvider {
    public:
      typedef std::string RecordName;
      typedef std::string DataType;
      typedef std::string DataLabel;
      typedef std::pair<DataType, DataLabel> DataKeyInfo;
      typedef std::multimap<RecordName, DataKeyInfo> RecordToDataMap;
      typedef std::map<ComponentDescription, RecordToDataMap> PreferredProviderInfo;

      EventSetupProvider(ActivityRegistry const*,
                         unsigned subProcessIndex = 0U,
                         PreferredProviderInfo const* iInfo = nullptr);
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
      void replaceExisting(std::shared_ptr<ESProductResolverProvider>);
      void add(std::shared_ptr<EventSetupRecordIntervalFinder>);

      void finishConfiguration(NumberOfConcurrentIOVs const&, bool& hasNonconcurrentFinder);

      ///Used when we need to force a Record to reset all its proxies
      void resetRecordPlusDependentRecords(EventSetupRecordKey const&);

      ///Used when testing that all code properly updates on IOV changes of all Records
      void forceCacheClear();

      void checkESProducerSharing(
          ModuleTypeResolverMaker const* resolverMaker,
          EventSetupProvider& precedingESProvider,
          std::set<ParameterSetIDHolder>& sharingCheckDone,
          std::map<EventSetupRecordKey, std::vector<ComponentDescription const*>>& referencedESProducers,
          EventSetupsController& esController);

      bool doRecordsMatch(EventSetupProvider& precedingESProvider,
                          EventSetupRecordKey const& eventSetupRecordKey,
                          std::map<EventSetupRecordKey, bool>& allComponentsMatch,
                          EventSetupsController const& esController);

      void fillReferencedDataKeys(EventSetupRecordKey const& eventSetupRecordKey);

      void resetRecordToResolverPointers();

      void clearInitializationData();

      unsigned subProcessIndex() const { return subProcessIndex_; }

      static void logInfoWhenSharing(ParameterSet const& iConfiguration);

      /// Intended for use only in tests
      void addRecord(const EventSetupRecordKey& iKey);
      void setPreferredProviderInfo(PreferredProviderInfo const& iInfo);

      void fillRecordsNotAllowingConcurrentIOVs(std::set<EventSetupRecordKey>& recordsNotAllowingConcurrentIOVs) const;

      void fillKeys(std::set<EventSetupRecordKey>& keys) const;

      EventSetupRecordProvider* tryToGetRecordProvider(const EventSetupRecordKey& iKey);

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

      bool mustFinishConfiguration_;
      unsigned subProcessIndex_;
      propagate_const<std::shared_ptr<EventSetupImpl>> eventSetupImpl_;

      // The following are all used only during initialization and then cleared.

      std::unique_ptr<PreferredProviderInfo> preferredProviderInfo_;
      std::unique_ptr<std::vector<std::shared_ptr<EventSetupRecordIntervalFinder>>> finders_;
      std::unique_ptr<std::vector<std::shared_ptr<ESProductResolverProvider>>> dataProviders_;
      std::unique_ptr<std::map<EventSetupRecordKey, std::map<DataKey, ComponentDescription const*>>> referencedDataKeys_;
      std::unique_ptr<std::map<EventSetupRecordKey, std::vector<std::shared_ptr<EventSetupRecordIntervalFinder>>>>
          recordToFinders_;
      std::unique_ptr<std::map<ParameterSetIDHolder, std::set<EventSetupRecordKey>>> psetIDToRecordKey_;
      std::unique_ptr<std::map<EventSetupRecordKey, std::map<DataKey, ComponentDescription>>> recordToPreferred_;
      std::unique_ptr<std::set<EventSetupRecordKey>> recordsWithALooperResolver_;
    };

  }  // namespace eventsetup
}  // namespace edm
#endif
