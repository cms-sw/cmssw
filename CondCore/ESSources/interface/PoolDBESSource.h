#ifndef CondCore_ESSources_PoolDBESSource_h
#define CondCore_ESSources_PoolDBESSource_h
//
// Package:    CondCore/ESSources
// Class:      PoolDBESSource
//
/**\class PoolDBESSource PoolDBESSource.h CondCore/ESSources/interface/PoolDBESSource.h
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
#include "CondCore/DBCommon/interface/DbConnection.h"

#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "CondCore/DBCommon/interface/TagMetadata.h"
#include "CondCore/DBCommon/interface/Time.h"
#include <boost/functional/hash.hpp>

namespace edm{
  class ParameterSet;
}
namespace cond{
  class DbSession;
  class BasePayloadProxy;
  class DataProxyWrapperBase;
}

class PoolDBESSource : public edm::eventsetup::DataProxyProvider,
		       public edm::EventSetupRecordIntervalFinder{
 public:
  typedef boost::shared_ptr<cond::DataProxyWrapperBase > ProxyP;
  typedef std::multimap< std::string,  ProxyP> ProxyMap;
 

  explicit PoolDBESSource( const edm::ParameterSet& );
  ~PoolDBESSource();
  
 protected:
  virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
			      const edm::IOVSyncValue& , 
			      edm::ValidityInterval&) ;

  virtual void registerProxies(const edm::eventsetup::EventSetupRecordKey& iRecordKey, KeyedProxies& aProxyList) ;

  virtual void newInterval(const edm::eventsetup::EventSetupRecordKey& iRecordType, const edm::ValidityInterval& iInterval) ;

 private:

  // ----------member data ---------------------------

  cond::DbConnection m_connection;

  // Container of DataProxy, implemented as multi-map keyed by records
  ProxyMap m_proxies;


  typedef std::set< cond::TagMetadata > TagCollection;
  // the collections of tag, record/label used in this ESSource
  TagCollection m_tagCollection;

  struct Stats {
    int nData;
    int nSet;
    int nRun;
    int nRefresh;
    int nActualRefresh;
  };

  Stats stats;

  unsigned int lastRun;
  bool doRefresh;

  bool doDump;

 private:

   void fillTagCollectionFromDB( cond::DbSession& coraldb,
				const std::string& roottag,
				std::map<std::string,cond::TagMetadata>& 
				 replacement);
};
#endif
