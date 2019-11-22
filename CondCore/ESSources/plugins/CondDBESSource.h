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

// Some comments on concurrency. There are 3 relevant things the Framework is
// doing that prevent concurrency issues. This happens automatically and
// CondDBESSource can rely on the Framework taking care of this.
//
//   1. There is a recursive global mutex which is locked while calls to
//   DataProxy::make and setIntervalFor are executing that allows only
//   1 thread to be running one of those functions at a time. (There
//   is some discussion about replacing this mutex with lockfree concurrency
//   mechanisms someday in the future, although this would be done in such
//   a way as to provide similar protection against data races.)
//
//   2. Calls are sequenced that a call to setIntervalFor is made, then
//   all related calls to DataProxy::initializeForNewIOV are made before
//   another call to setIntervalFor is made.  It is configurable how many
//   IOVs can be running concurrently. The Framework will not call
//   initializeForNewIOV or start running a new IOV unless the
//   number of active IOVs is less than that configured number.
//
//   3. Independent of the above two items, after a call is made to
//   DataProxy::initializeForNewIOV for a particular
//   EventSetupRecordKey and iovIndex, all calls to DataProxy::make
//   associated with that whose data is requested will be completed
//   and processing of luminosity blocks associated with that will
//   be completed before another call to DataProxy::initializeForNewIOV
//   is made for that EventSetupRecordKey and iovIndex.

// system include files
#include <string>
#include <map>
#include <memory>
#include <set>
// user include files
#include "CondCore/CondDB/interface/ConnectionPool.h"

#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
//#include "CondCore/DBCommon/interface/Time.h"

namespace edm {
  class ParameterSet;
}

namespace cond {
  class DataProxyWrapperBase;
}

class CondDBESSource : public edm::eventsetup::DataProxyProvider, public edm::EventSetupRecordIntervalFinder {
public:
  using DataKey = edm::eventsetup::DataKey;
  using EventSetupRecordKey = edm::eventsetup::EventSetupRecordKey;
  typedef std::shared_ptr<cond::DataProxyWrapperBase> ProxyP;
  typedef std::multimap<std::string, ProxyP> ProxyMap;

  typedef enum { NOREFRESH, REFRESH_ALWAYS, REFRESH_OPEN_IOVS, REFRESH_EACH_RUN, RECONNECT_EACH_RUN } RefreshPolicy;

  explicit CondDBESSource(const edm::ParameterSet&);
  ~CondDBESSource() override;

protected:
  void setIntervalFor(const EventSetupRecordKey&, const edm::IOVSyncValue&, edm::ValidityInterval&) override;

  KeyedProxiesVector registerProxies(const EventSetupRecordKey&, unsigned int iovIndex) override;

  void initConcurrentIOVs(const EventSetupRecordKey& key, unsigned int nConcurrentIOVs) override;

  bool isConcurrentFinder() const override { return true; }

private:
  // ----------member data ---------------------------

  cond::persistency::ConnectionPool m_connection;
  std::string m_connectionString;

  // Container of DataProxy, implemented as multi-map keyed by records
  ProxyMap m_proxies;

  typedef std::map<std::string, cond::GTEntry_t> TagCollection;
  // the collections of tag, record/label used in this ESSource
  TagCollection m_tagCollection;
  std::map<std::string, std::pair<cond::persistency::Session, std::string> > m_sessionPool;
  std::map<std::string, unsigned int> m_lastRecordRuns;

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
  void fillList(const std::string& pfn,
                std::vector<std::string>& pfnList,
                const unsigned int listSize,
                const std::string& type);

  void fillTagCollectionFromGT(const std::string& connectionString,
                               const std::string& prefix,
                               const std::string& postfix,
                               const std::string& roottag,
                               std::set<cond::GTEntry_t>& tagcoll,
                               cond::GTMetadata_t& gtMetadata);

  void fillTagCollectionFromDB(const std::vector<std::string>& connectionStringList,
                               const std::vector<std::string>& prefixList,
                               const std::vector<std::string>& postfixList,
                               const std::vector<std::string>& roottagList,
                               std::map<std::string, cond::GTEntry_t>& replacement,
                               cond::GTMetadata_t& gtMetadata);
};
#endif
