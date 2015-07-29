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
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupKnownRecordsSupplier.h"

// system include files
#include "boost/shared_ptr.hpp"

#include <memory>
#include <map>
#include <set>
#include <string>
#include <vector>


// forward declarations
namespace edm {
   class EventSetupRecordIntervalFinder;
   class IOVSyncValue;
   class ParameterSet;

   namespace eventsetup {
      struct ComponentDescription;
      class DataKey;
      class DataProxyProvider;
      class EventSetupRecord;
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

      EventSetupProvider(unsigned subProcessIndex = 0U, PreferredProviderInfo const* iInfo = 0);
      virtual ~EventSetupProvider();

      // ---------- const member functions ---------------------
      std::set<ComponentDescription> proxyProviderDescriptions() const;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      EventSetup const& eventSetupForInstance(IOVSyncValue const&);

      EventSetup const& eventSetup() const {return eventSetup_;}

      //called by specializations of EventSetupRecordProviders
      void addRecordToEventSetup(EventSetupRecord& iRecord);

      void add(boost::shared_ptr<DataProxyProvider>);
      void replaceExisting(boost::shared_ptr<DataProxyProvider>);
      void add(boost::shared_ptr<EventSetupRecordIntervalFinder>);

      void finishConfiguration();

      ///Used when we need to force a Record to reset all its proxies
      void resetRecordPlusDependentRecords(EventSetupRecordKey const&);

      ///Used when testing that all code properly updates on IOV changes of all Records
      void forceCacheClear();

      void checkESProducerSharing(EventSetupProvider & precedingESProvider,
                                  std::set<ParameterSetIDHolder>& sharingCheckDone,
                                  std::map<EventSetupRecordKey, std::vector<ComponentDescription const*> >& referencedESProducers,
                                  EventSetupsController & esController);

      bool doRecordsMatch(EventSetupProvider & precedingESProvider,
                          EventSetupRecordKey const& eventSetupRecordKey,
                          std::map<EventSetupRecordKey, bool> & allComponentsMatch,
                          EventSetupsController const& esController);

      void fillReferencedDataKeys(EventSetupRecordKey const& eventSetupRecordKey);

      void resetRecordToProxyPointers();

      void clearInitializationData();

      unsigned subProcessIndex() const { return subProcessIndex_; }

      static void logInfoWhenSharing(ParameterSet const& iConfiguration);

   protected:

      template <typename T>
         void insert(std::auto_ptr<T> iRecordProvider) {
            std::auto_ptr<EventSetupRecordProvider> temp(iRecordProvider.release());
            insert(eventsetup::heterocontainer::makeKey<
                    typename T::RecordType,
                       eventsetup::EventSetupRecordKey>(),
                    temp);
         }

   private:
      EventSetupProvider(EventSetupProvider const&); // stop default

      EventSetupProvider const& operator=(EventSetupProvider const&); // stop default

      void insert(EventSetupRecordKey const&, std::auto_ptr<EventSetupRecordProvider>);

      // ---------- member data --------------------------------
      EventSetup eventSetup_;
      typedef std::map<EventSetupRecordKey, boost::shared_ptr<EventSetupRecordProvider> > Providers;
      Providers providers_;
      std::unique_ptr<EventSetupKnownRecordsSupplier> knownRecordsSupplier_;
      bool mustFinishConfiguration_;
      unsigned subProcessIndex_;

      // The following are all used only during initialization and then cleared.

      std::unique_ptr<PreferredProviderInfo> preferredProviderInfo_;
      std::unique_ptr<std::vector<boost::shared_ptr<EventSetupRecordIntervalFinder> > > finders_;
      std::unique_ptr<std::vector<boost::shared_ptr<DataProxyProvider> > > dataProviders_;
      std::unique_ptr<std::map<EventSetupRecordKey, std::map<DataKey, ComponentDescription const*> > > referencedDataKeys_;
      std::unique_ptr<std::map<EventSetupRecordKey, std::vector<boost::shared_ptr<EventSetupRecordIntervalFinder> > > > recordToFinders_;
      std::unique_ptr<std::map<ParameterSetIDHolder, std::set<EventSetupRecordKey> > > psetIDToRecordKey_;
      std::unique_ptr<std::map<EventSetupRecordKey, std::map<DataKey, ComponentDescription> > > recordToPreferred_;
      std::unique_ptr<std::set<EventSetupRecordKey> > recordsWithALooperProxy_;
};

   }
}
#endif
