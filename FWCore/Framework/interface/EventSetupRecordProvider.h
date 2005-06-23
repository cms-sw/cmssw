#ifndef EVENTSETUP_EVENTSETUPRECORDPROVIDER_H
#define EVENTSETUP_EVENTSETUPRECORDPROVIDER_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     EventSetupRecordProvider
// 
/**\class EventSetupRecordProvider EventSetupRecordProvider.h Core/CoreFramework/interface/EventSetupRecordProvider.h

 Description: Coordinates all EventSetupDataProviders with the same 'interval of validity'

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Fri Mar 25 19:02:23 EST 2005
//

// system include files
#include <vector>
#include <set>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/CoreFramework/interface/ValidityInterval.h"
#include "FWCore/CoreFramework/interface/EventSetupRecordKey.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      class EventSetupProvider;
      class DataProxyProvider;
      class EventSetupRecordIntervalFinder;
      
class EventSetupRecordProvider
{

   public:
      EventSetupRecordProvider(const EventSetupRecordKey& iKey);
      virtual ~EventSetupRecordProvider();

      // ---------- const member functions ---------------------

      const ValidityInterval& validityInterval() const {
         return validityInterval_;
      }
      const EventSetupRecordKey& key() const { return key_; }      

      ///Returns the list of Records the provided Record depends on (usually none)
      virtual std::set<EventSetupRecordKey> dependentRecords() const;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void addRecordTo(EventSetupProvider&) = 0;
      void addRecordToIfValid(EventSetupProvider&, const Timestamp&) ;

      void add(boost::shared_ptr<DataProxyProvider>);
      ///For now, only use one finder
      void addFinder(boost::shared_ptr<EventSetupRecordIntervalFinder>);
      void setValidityInterval(const ValidityInterval&);
      
      ///sets interval to this time and returns true if have a valid interval for time
      bool setValidityIntervalFor(const Timestamp&);

      ///If the provided Record depends on other Records, here are the dependent Providers
      virtual void setDependentProviders(const std::vector< boost::shared_ptr<EventSetupRecordProvider> >&);

   protected:
      virtual void addProxiesToRecord(boost::shared_ptr<DataProxyProvider>) = 0;
   private:
      EventSetupRecordProvider(const EventSetupRecordProvider&); // stop default

      const EventSetupRecordProvider& operator=(const EventSetupRecordProvider&); // stop default

      // ---------- member data --------------------------------
      const EventSetupRecordKey key_;
      ValidityInterval validityInterval_;
      boost::shared_ptr<EventSetupRecordIntervalFinder> finder_;
      std::vector< boost::shared_ptr<DataProxyProvider> > providers_;
};
   }
}

#endif /* EVENTSETUP_EVENTSETUPRECORDPROVIDER_H */
