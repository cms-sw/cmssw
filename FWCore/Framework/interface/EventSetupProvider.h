#ifndef FWCore_Framework_EventSetupProvider_h
#define FWCore_Framework_EventSetupProvider_h
// -*- C++ -*-
//
// Package:     Framework
// Class:      EventSetupProvider
//
/**\class EventSetupProvider EventSetupProvider.h FWCore/Framework/interface/EventSetupProvider.h

 Description: Factory for a EventSetup

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu Mar 24 14:10:07 EST 2005
//

// user include files
#include "FWCore/Framework/interface/EventSetupImpl.h"

// system include files

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

// forward declarations
namespace edm {
  class ActivityRegistry;
  class EventSetupRecordIntervalFinder;
  class IOVSyncValue;
  class ParameterSet;

  namespace eventsetup {
    struct ComponentDescription;
    class DataKey;
    class DataProxyProvider;
    class EventSetupRecordImpl;
    class EventSetupRecordKey;
    class EventSetupRecordProvider;
    class EventSetupsController;
    class ParameterSetIDHolder;

    class EventSetupProvider {
    public:
      typedef std::string RecordName;
      typedef std::string DataType;
      typedef std::string DataLabel;
      typedef std::pair<DataType, DataLabel> DataKeyInfo;
      typedef std::multimap<RecordName, DataKeyInfo> RecordToDataMap;
      typedef std::map<ComponentDescription, RecordToDataMap> PreferredProviderInfo;

      EventSetupProvider(ActivityRegistry*,
                         unsigned subProcessIndex = 0U,
                         PreferredProviderInfo const* iInfo = nullptr);
      virtual ~EventSetupProvider();

      // ---------- const member functions ---------------------
      std::set<ComponentDescription> proxyProviderDescriptions() const;
      bool isWithinValidityInterval(IOVSyncValue const&) const;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      EventSetupImpl const& eventSetupForInstance(IOVSyncValue const&);

      EventSetupImpl const& eventSetup() const { return eventSetup_; }

      //called by specializations of EventSetupRecordProviders
      void addRecordToEventSetup(EventSetupRecordImpl& iRecord);

      void add(std::shared_ptr<DataProxyProvider>);
      void replaceExisting(std::shared_ptr<DataProxyProvider>);
      void add(std::shared_ptr<EventSetupRecordIntervalFinder>);

      void finishConfiguration();

      ///Used when we need to force a Record to reset all its proxies
      void resetRecordPlusDependentRecords(EventSetupRecordKey const&);

      ///Used when testing that all code properly updates on IOV changes of all Records
      void forceCacheClear();

      void checkESProducerSharing(
          EventSetupProvider& precedingESProvider,
          std::set<ParameterSetIDHolder>& sharingCheckDone,
          std::map<EventSetupRecordKey, std::vector<ComponentDescription const*>>& referencedESProducers,
          EventSetupsController& esController);

      bool doRecordsMatch(EventSetupProvider& precedingESProvider,
                          EventSetupRecordKey const& eventSetupRecordKey,
                          std::map<EventSetupRecordKey, bool>& allComponentsMatch,
                          EventSetupsController const& esController);

      void fillReferencedDataKeys(EventSetupRecordKey const& eventSetupRecordKey);

      void resetRecordToProxyPointers();

      void clearInitializationData();

      unsigned subProcessIndex() const { return subProcessIndex_; }

      static void logInfoWhenSharing(ParameterSet const& iConfiguration);

      /// Intended for use only in tests
      void addRecord(const EventSetupRecordKey& iKey);

    protected:
      void insert(std::unique_ptr<EventSetupRecordProvider> iRecordProvider);

    private:
      EventSetupProvider(EventSetupProvider const&) = delete;  // stop default

      EventSetupProvider const& operator=(EventSetupProvider const&) = delete;  // stop default

      std::shared_ptr<EventSetupRecordProvider>& recordProvider(const EventSetupRecordKey& iKey);
      EventSetupRecordProvider* tryToGetRecordProvider(const EventSetupRecordKey& iKey);
      void insert(EventSetupRecordKey const&, std::unique_ptr<EventSetupRecordProvider>);

      void determinePreferred();

      // ---------- member data --------------------------------
      EventSetupImpl eventSetup_;

      using RecordKeys = std::vector<EventSetupRecordKey>;
      RecordKeys recordKeys_;

      using RecordProviders = std::vector<std::shared_ptr<EventSetupRecordProvider>>;
      RecordProviders recordProviders_;

      bool mustFinishConfiguration_;
      unsigned subProcessIndex_;

      // The following are all used only during initialization and then cleared.

      std::unique_ptr<PreferredProviderInfo> preferredProviderInfo_;
      std::unique_ptr<std::vector<std::shared_ptr<EventSetupRecordIntervalFinder>>> finders_;
      std::unique_ptr<std::vector<std::shared_ptr<DataProxyProvider>>> dataProviders_;
      std::unique_ptr<std::map<EventSetupRecordKey, std::map<DataKey, ComponentDescription const*>>> referencedDataKeys_;
      std::unique_ptr<std::map<EventSetupRecordKey, std::vector<std::shared_ptr<EventSetupRecordIntervalFinder>>>>
          recordToFinders_;
      std::unique_ptr<std::map<ParameterSetIDHolder, std::set<EventSetupRecordKey>>> psetIDToRecordKey_;
      std::unique_ptr<std::map<EventSetupRecordKey, std::map<DataKey, ComponentDescription>>> recordToPreferred_;
      std::unique_ptr<std::set<EventSetupRecordKey>> recordsWithALooperProxy_;
    };

  }  // namespace eventsetup
}  // namespace edm
#endif
