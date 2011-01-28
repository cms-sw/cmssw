#ifndef FWCore_Framework_EventSetupRecordProvider_h
#define FWCore_Framework_EventSetupRecordProvider_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordProvider
// 
/**\class EventSetupRecordProvider EventSetupRecordProvider.h FWCore/Framework/interface/EventSetupRecordProvider.h

 Description: Coordinates all EventSetupDataProviders with the same 'interval of validity'

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Fri Mar 25 19:02:23 EST 2005
//

// user include files
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/ValidityInterval.h"

// system include files
#include "boost/shared_ptr.hpp"

#include <map>
#include <memory>
#include <set>
#include <vector>

// forward declarations
namespace edm {
   class EventSetupRecordIntervalFinder;

   namespace eventsetup {
      struct ComponentDescription;
      class DataKey;
      class DataProxyProvider;
      class EventSetupProvider;
      
class EventSetupRecordProvider {

   public:
      typedef std::map<DataKey, ComponentDescription> DataToPreferredProviderMap;
   
      EventSetupRecordProvider(EventSetupRecordKey const& iKey);
      virtual ~EventSetupRecordProvider();

      // ---------- const member functions ---------------------

      ValidityInterval const& validityInterval() const {
         return validityInterval_;
      }
      EventSetupRecordKey const& key() const { return key_; }      

      ///Returns the list of Records the provided Record depends on (usually none)
      virtual std::set<EventSetupRecordKey> dependentRecords() const;
      
      ///return information on which DataProxyProviders are supplying information
      std::set<ComponentDescription> proxyProviderDescriptions() const;
      
      ///returns the DataProxyProvider or a 'null' if not found
      boost::shared_ptr<DataProxyProvider> proxyProvider(ComponentDescription const&) const;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void addRecordTo(EventSetupProvider&) = 0;
      void addRecordToIfValid(EventSetupProvider&, IOVSyncValue const&) ;

      void add(boost::shared_ptr<DataProxyProvider>);
      ///For now, only use one finder
      void addFinder(boost::shared_ptr<EventSetupRecordIntervalFinder>);
      void setValidityInterval(ValidityInterval const&);
      
      ///sets interval to this time and returns true if have a valid interval for time
      bool setValidityIntervalFor(IOVSyncValue const&);

      ///If the provided Record depends on other Records, here are the dependent Providers
      virtual void setDependentProviders(std::vector<boost::shared_ptr<EventSetupRecordProvider> >const&);

      /**In the case of a conflict, sets what Provider to call.  This must be called after
         all providers have been added.  An empty map is acceptable. */
      void usePreferred(DataToPreferredProviderMap const&);
      
      ///This will clear the cache's of all the Proxies so that next time they are called they will run
      void resetProxies();
      
      boost::shared_ptr<EventSetupRecordIntervalFinder> finder() const { return finder_; }

   protected:
      virtual void addProxiesToRecord(boost::shared_ptr<DataProxyProvider>,
                                      DataToPreferredProviderMap const&) = 0;
      virtual void cacheReset() = 0;
      
      boost::shared_ptr<EventSetupRecordIntervalFinder> swapFinder(boost::shared_ptr<EventSetupRecordIntervalFinder> iNew) {
        std::swap(iNew, finder_);
        return iNew;
      }

   private:
      EventSetupRecordProvider(EventSetupRecordProvider const&); // stop default

      EventSetupRecordProvider const& operator=(EventSetupRecordProvider const&); // stop default

      void resetTransients();
      virtual bool checkResetTransients() = 0;
      // ---------- member data --------------------------------
      EventSetupRecordKey const key_;
      ValidityInterval validityInterval_;
      boost::shared_ptr<EventSetupRecordIntervalFinder> finder_;
      std::vector<boost::shared_ptr<DataProxyProvider> > providers_;
      std::auto_ptr<std::vector<boost::shared_ptr<EventSetupRecordIntervalFinder> > > multipleFinders_;
      bool lastSyncWasBeginOfRun_;
};
   }
}

#endif
