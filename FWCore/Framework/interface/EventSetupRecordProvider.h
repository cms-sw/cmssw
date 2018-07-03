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
#include "FWCore/Framework/interface/EventSetupRecordImpl.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

// system include files

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
      class EventSetupRecordImpl;
      class ParameterSetIDHolder;
      
class EventSetupRecordProvider {

   public:
      typedef std::map<DataKey, ComponentDescription> DataToPreferredProviderMap;
   
      EventSetupRecordProvider(EventSetupRecordKey const& iKey);

      // ---------- const member functions ---------------------

      ValidityInterval const& validityInterval() const {
         return validityInterval_;
      }
      EventSetupRecordKey const& key() const { return key_; }      

      EventSetupRecordImpl const& record() const {return record_;}
      EventSetupRecordImpl& record() { return record_;}

      ///Returns the list of Records the provided Record depends on (usually none)
      std::set<EventSetupRecordKey> dependentRecords() const;
      
      ///return information on which DataProxyProviders are supplying information
      std::set<ComponentDescription> proxyProviderDescriptions() const;
  
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      ///returns the first matching DataProxyProvider or a 'null' if not found
      std::shared_ptr<DataProxyProvider> proxyProvider(ComponentDescription const&);
  
      ///returns the first matching DataProxyProvider or a 'null' if not found
      std::shared_ptr<DataProxyProvider> proxyProvider(ParameterSetIDHolder const&);
  

      void resetProxyProvider(ParameterSetIDHolder const&, std::shared_ptr<DataProxyProvider> const&);

      void addRecordTo(EventSetupProvider&);
      void addRecordToIfValid(EventSetupProvider&, IOVSyncValue const&) ;

      void add(std::shared_ptr<DataProxyProvider>);
      ///For now, only use one finder
      void addFinder(std::shared_ptr<EventSetupRecordIntervalFinder>);
      void setValidityInterval(ValidityInterval const&);
      
      ///sets interval to this time and returns true if have a valid interval for time
      bool setValidityIntervalFor(IOVSyncValue const&);

      ///If the provided Record depends on other Records, here are the dependent Providers
      void setDependentProviders(std::vector<std::shared_ptr<EventSetupRecordProvider> >const&);

      /**In the case of a conflict, sets what Provider to call.  This must be called after
         all providers have been added.  An empty map is acceptable. */
      void usePreferred(DataToPreferredProviderMap const&);
      
      ///This will clear the cache's of all the Proxies so that next time they are called they will run
      void resetProxies();
      
      std::shared_ptr<EventSetupRecordIntervalFinder const> finder() const {return get_underlying_safe(finder_);}
      std::shared_ptr<EventSetupRecordIntervalFinder>& finder() {return get_underlying_safe(finder_);}

      void getReferencedESProducers(std::map<EventSetupRecordKey, std::vector<ComponentDescription const*> >& referencedESProducers);

      void fillReferencedDataKeys(std::map<DataKey, ComponentDescription const*>& referencedDataKeys);

      void resetRecordToProxyPointers(DataToPreferredProviderMap const& iMap);

   protected:
      void addProxiesToRecordHelper(edm::propagate_const<std::shared_ptr<DataProxyProvider>>& dpp,
                              DataToPreferredProviderMap const& mp) {addProxiesToRecord(get_underlying_safe(dpp), mp);}
      void addProxiesToRecord(std::shared_ptr<DataProxyProvider>,
                              DataToPreferredProviderMap const&);
      void cacheReset();

      std::shared_ptr<EventSetupRecordIntervalFinder> swapFinder(std::shared_ptr<EventSetupRecordIntervalFinder> iNew) {
        std::swap(iNew, finder());
        return iNew;
      }

   private:
      EventSetupRecordProvider(EventSetupRecordProvider const&) = delete; // stop default

      EventSetupRecordProvider const& operator=(EventSetupRecordProvider const&) = delete; // stop default

      void resetTransients();
      bool checkResetTransients();
      // ---------- member data --------------------------------
      EventSetupRecordImpl record_;
      EventSetupRecordKey const key_;
      ValidityInterval validityInterval_;
      edm::propagate_const<std::shared_ptr<EventSetupRecordIntervalFinder>> finder_;
      std::vector<edm::propagate_const<std::shared_ptr<DataProxyProvider>>> providers_;
      std::unique_ptr<std::vector<edm::propagate_const<std::shared_ptr<EventSetupRecordIntervalFinder>>>> multipleFinders_;
      bool lastSyncWasBeginOfRun_;
};
   }
}

#endif
