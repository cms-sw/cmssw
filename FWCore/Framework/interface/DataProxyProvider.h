#ifndef EVENTSETUP_DATAPROXYPROVIDER_H
#define EVENTSETUP_DATAPROXYPROVIDER_H
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
#include <vector>
#include <set>
#include <map>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/DataKey.h"

// forward declarations
namespace edm {
   class ValidityInterval;
   namespace eventsetup {
      class DataProxy;
class DataProxyProvider
{

   public:
      typedef std::vector< EventSetupRecordKey> Keys;
      typedef std::vector<std::pair<DataKey, 
                                    boost::shared_ptr<DataProxy> > > KeyedProxies ;
      typedef std::map<EventSetupRecordKey, KeyedProxies> RecordProxies;
      
      DataProxyProvider();
      virtual ~DataProxyProvider();

      // ---------- const member functions ---------------------
      bool isUsingRecord(const EventSetupRecordKey&) const;
      
      std::set<EventSetupRecordKey> usingRecords() const;
      
      const KeyedProxies& keyedProxies(const EventSetupRecordKey& iRecordKey) const ;
      
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      
      ///called when a new interval of validity occurs for iRecordType
      virtual void newInterval(const EventSetupRecordKey& iRecordType,
                                const ValidityInterval& iInterval) = 0;
      
   protected:
      template< class T>
      void usingRecord() {
         usingRecordWithKey(EventSetupRecordKey::makeKey<T>());
      }
      
      void usingRecordWithKey(const EventSetupRecordKey&);

      void invalidateProxies(const EventSetupRecordKey& iRecordKey) ;

      virtual void registerProxies(const EventSetupRecordKey& iRecordKey ,
                                    KeyedProxies& aProxyList) = 0 ;
      
      ///deletes all the Proxies in aStream
      void eraseAll(const EventSetupRecordKey& iRecordKey) ;

   private:
      DataProxyProvider(const DataProxyProvider&); // stop default

      const DataProxyProvider& operator=(const DataProxyProvider&); // stop default

      // ---------- member data --------------------------------
      RecordProxies recordProxies_;
};

template<class ProxyT>
inline void insertProxy(DataProxyProvider::KeyedProxies& iList,
                        boost::shared_ptr<ProxyT> iProxy,
                        const char* iName="") {
   iList.push_back(DataProxyProvider::KeyedProxies::value_type(
                                             DataKey(DataKey::makeTypeTag<typename ProxyT::value_type>(),
                                                     iName),
                                             iProxy));
   
}

   }
}
#endif /* EVENTSETUP_DATAPROXYPROVIDER_H */
