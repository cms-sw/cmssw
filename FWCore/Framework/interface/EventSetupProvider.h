#ifndef Framework_EventSetupProvider_h
#define Framework_EventSetupProvider_h
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

// system include files
#include <memory>
#include <map>
#include <set>

#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/EventSetup.h"

// forward declarations
namespace edm {
   class IOVSyncValue;
   class EventSetupRecordIntervalFinder;
   
   namespace eventsetup {
      class EventSetupRecordProvider;
      class DataProxyProvider;
      class ComponentDescription;
      
class EventSetupProvider
{

   public:
      typedef std::string RecordName;
      typedef std::string DataType;
      typedef std::string DataLabel;
      typedef std::pair<DataType,DataLabel> DataKeyInfo;
      typedef std::multimap<RecordName,DataKeyInfo> RecordToDataMap;
      typedef std::map<ComponentDescription,RecordToDataMap > PreferredProviderInfo;
      EventSetupProvider(const PreferredProviderInfo* iInfo=0);
      virtual ~EventSetupProvider();

      // ---------- const member functions ---------------------
      std::set<ComponentDescription> proxyProviderDescriptions() const;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      EventSetup const& eventSetupForInstance(const IOVSyncValue&);

      EventSetup const& eventSetup() const {return eventSetup_;}
     
      //called by specializations of EventSetupRecordProviders
      template<class T>
         void addRecordToEventSetup(T& iRecord) {
            iRecord.setEventSetup(&eventSetup_);
            eventSetup_.add(iRecord);
         }
      
      void add(boost::shared_ptr<DataProxyProvider>);
      void add(boost::shared_ptr<EventSetupRecordIntervalFinder>);
      
      void finishConfiguration();
      
      ///Used when we need to force a Record to reset all its proxies
      void resetRecordPlusDependentRecords(const EventSetupRecordKey& );
   
      ///Used when testing that all code properly updates on IOV changes of all Records
      void forceCacheClear();
   
   protected:

      template <class T>
         void insert(std::auto_ptr<T> iRecordProvider) {
            std::auto_ptr<EventSetupRecordProvider> temp(iRecordProvider.release());
            insert(eventsetup::heterocontainer::makeKey<
                    typename T::RecordType, 
                       eventsetup::EventSetupRecordKey>(),
                    temp);
         }
      
   private:
      EventSetupProvider(const EventSetupProvider&); // stop default

      const EventSetupProvider& operator=(const EventSetupProvider&); // stop default


      void insert(const EventSetupRecordKey&, std::auto_ptr<EventSetupRecordProvider>);
      
      // ---------- member data --------------------------------
      EventSetup eventSetup_;
      typedef std::map<EventSetupRecordKey, boost::shared_ptr<EventSetupRecordProvider> > Providers;
      Providers providers_;
      bool mustFinishConfiguration_;
      
      std::auto_ptr<PreferredProviderInfo> preferredProviderInfo_;
      std::auto_ptr<std::vector<boost::shared_ptr<EventSetupRecordIntervalFinder> > > finders_;
      std::auto_ptr<std::vector<boost::shared_ptr<DataProxyProvider> > > dataProviders_;
};

   }
}
#endif
