#ifndef CondCore_ESSources_CondDBESSource_h
#define CondCore_ESSources_CondDBESSource_h
//
// Package:    CondCore/ESSources
// Class:      CondDBESSource
//
/*
 Description: EventSetup source module for serving data from offline database
*/
//
// Author:      Zhen Xie
//
// system include files
#include <string>
#include <map>
#include <set>
// user include files
#include "CondCore/CondDB/interface/ConnectionPool.h"

#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
//#include "CondCore/DBCommon/interface/Time.h"

namespace edm{
  class ParameterSet;
}

namespace cond{
  class DataProxyWrapperBase;
}

class CondDBESSource : public edm::eventsetup::DataProxyProvider,
		       public edm::EventSetupRecordIntervalFinder{
 public:
  typedef boost::shared_ptr<cond::DataProxyWrapperBase > ProxyP;
  typedef std::multimap< std::string,  ProxyP> ProxyMap;

  typedef enum { NOREFRESH, REFRESH_ALWAYS, REFRESH_OPEN_IOVS, REFRESH_EACH_RUN, RECONNECT_EACH_RUN } RefreshPolicy;
  

  explicit CondDBESSource( const edm::ParameterSet& );
  ~CondDBESSource();
  
 protected:
  virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
			      const edm::IOVSyncValue& , 
			      edm::ValidityInterval&) ;

  virtual void registerProxies(const edm::eventsetup::EventSetupRecordKey& iRecordKey, KeyedProxies& aProxyList) ;

  virtual void newInterval(const edm::eventsetup::EventSetupRecordKey& iRecordType, const edm::ValidityInterval& iInterval) ;

 private:

  // ----------member data ---------------------------

  cond::persistency::ConnectionPool m_connection;
  std::string m_connectionString;

  // Container of DataProxy, implemented as multi-map keyed by records
  ProxyMap m_proxies;


  typedef std::map< std::string, cond::GTEntry_t > TagCollection;
  // the collections of tag, record/label used in this ESSource
  TagCollection m_tagCollection;
  std::map<std::string,std::pair<cond::persistency::Session,std::string> > m_sessionPool;
  std::map<std::string,unsigned int> m_lastRecordRuns;
  
  struct Stats {
    int nData;
    int nSet;
    int nRun;
    int nLumi;
    int nRefresh;
    int nActualRefresh;
    int nReconnect;
    int nActualReconnect;
  };

  Stats m_stats;

  unsigned int m_lastRun;
  unsigned int m_lastLumi;
  RefreshPolicy m_policy;
  
  bool m_doDump;

 private:

  void fillList(const std::string & pfn, std::vector<std::string> & pfnList, const unsigned int listSize, const std::string & type);

  void fillTagCollectionFromGT(const std::string & connectionString,
                               const std::string & prefix,
                               const std::string & postfix,
                               const std::string & roottag,
                               std::set< cond::GTEntry_t > & tagcoll,
			       cond::GTMetadata_t& gtMetadata);

  void fillTagCollectionFromDB( const std::vector<std::string> & connectionStringList,
                                const std::vector<std::string> & prefixList,
                                const std::vector<std::string> & postfixList,
                                const std::vector<std::string> & roottagList,
                                std::map<std::string,cond::GTEntry_t>& replacement,
				cond::GTMetadata_t& gtMetadata);
};
#endif
