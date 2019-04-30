#ifndef Framework_DataProxyProvider_h
#define Framework_DataProxyProvider_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DataProxyProvider
// 
/**\class DataProxyProvider DataProxyProvider.h FWCore/Framework/interface/DataProxyProvider.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Mon Mar 28 14:21:58 EST 2005
//

// system include files
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Utilities/interface/propagate_const.h"

// forward declarations
namespace edm {
   class ValidityInterval;
   class ParameterSet;
   class ConfigurationDescriptions;
   namespace eventsetup {
      class DataProxy;
      class ESRecordsToProxyIndices;
      
class DataProxyProvider
{

   public:   
      typedef std::vector< EventSetupRecordKey> Keys;
      typedef std::vector<std::pair<DataKey, edm::propagate_const<std::shared_ptr<DataProxy>>>> KeyedProxies;
      typedef std::map<EventSetupRecordKey, std::vector<KeyedProxies>> RecordProxies;
      
      DataProxyProvider();
      DataProxyProvider(const DataProxyProvider&) = delete;
      const DataProxyProvider& operator=(const DataProxyProvider&) = delete;
      virtual ~DataProxyProvider() noexcept(false);

      // ---------- const member functions ---------------------
      bool isUsingRecord(const EventSetupRecordKey&) const;
      
      std::set<EventSetupRecordKey> usingRecords() const;
      
      KeyedProxies& keyedProxies(const EventSetupRecordKey& iRecordKey, unsigned int iovIndex = 0);
      
      const ComponentDescription& description() const { return description_;}
      // ---------- static member functions --------------------
      /**Used to add parameters available to all inheriting classes
      */
      static void prevalidate(ConfigurationDescriptions&);

      // ---------- member functions ---------------------------

      virtual void updateLookup(ESRecordsToProxyIndices const&);

      void setDescription(const ComponentDescription& iDescription) {
         description_ = iDescription;
      }
      
      /**This method is only to be called by the framework, it sets the string
        which will be appended to the labels of all data products being produced
      **/
      void setAppendToDataLabel(const edm::ParameterSet&);

      void fillRecordsNotAllowingConcurrentIOVs(std::set<EventSetupRecordKey>& recordsNotAllowingConcurrentIOVs) const;

      void resizeKeyedProxiesVector(EventSetupRecordKey const& key, unsigned int nConcurrentIOVs);

   protected:
      template< class T>
      void usingRecord() {
         usingRecordWithKey(EventSetupRecordKey::makeKey<T>());
      }
      
      void usingRecordWithKey(const EventSetupRecordKey&);

      virtual void registerProxies(const EventSetupRecordKey& iRecordKey,
                                   KeyedProxies& aProxyList,
                                   unsigned int iovIndex) = 0;

      ///deletes all the Proxies in aStream
      void eraseAll(const EventSetupRecordKey& iRecordKey) ;

   private:

      // ---------- member data --------------------------------
      RecordProxies recordProxies_;
      ComponentDescription description_;
      std::string appendToDataLabel_;
};

template<class ProxyT>
inline void insertProxy(DataProxyProvider::KeyedProxies& iList,
                        std::shared_ptr<ProxyT> iProxy,
                        const char* iName="") {
   iList.push_back(DataProxyProvider::KeyedProxies::value_type(
                                             DataKey(DataKey::makeTypeTag<typename ProxyT::value_type>(),
                                                     iName),
                                             iProxy));
   
}

   }
}
#endif
