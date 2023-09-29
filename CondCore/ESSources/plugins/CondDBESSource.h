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

// Some comments on concurrency. Several things are working together
// to prevent concurrency issues when this module is executing.
// Some of these things are in this module and some are in the Framework.
// Here is a list of these things:
//
//   1. There is a single mutex that is a data member of CondDBESSource.
//   This is locked near the beginning of setIntervalFor and also
//   near the beginning of ::ProductResolver::prefetch so that these functions
//   will never run concurrently. All the ::ProductResolver objects have a
//   pointer to this mutex stored in their ESSourceProductResolverTemplate
//   base class.
//
//   2. CondDBESSource contains a single SerialTaskQueue. The tasks
//   that run the prefetch function are placed in this SerialTaskQueue.
//   This allows only one ::ProductResolver::prefetch function to run at a
//   time. All the ::ProductResolver objects have a pointer to this SerialTaskQueue
//   stored in their ESSourceProductResolverTemplate base class. Note that
//   locking the mutex is inside the task that runs prefetch.
//   Since these tasks are serialized by the SerialTaskQueue,
//   the mutex will never be locked by another prefetch call
//   when prefetch is called. The mutex is really only protecting
//   setIntervalFor calls from each other and from prefetch calls.
//
//   3. An ESSource is not allowed to get data from the EventSetup
//   while its ProductResolver prefetch function runs, preventing deadlocks
//   and ensuring the mutex does not need to be recursive.
//
//   4. The WaitingTaskList in ESSourceProductResolverBase (a base class of
//   ::ProductResolver) is used to notify other tasks waiting for prefetch
//   to complete that the data is available (other tasks created and
//   managed by the Framework).
//
//   5. There is an atomic<bool> in ESSourceProductResolverBase which
//   prevents the prefetch function being run more than once for the
//   same IOV and ProductResolver.
//
//   6. The Framework ensures calls are sequenced such that a call to
//   setIntervalFor is made and completes, then all related calls to
//   ProductResolver::initializeForNewIOV are made before another call to
//   setIntervalFor is made.  It is configurable how many
//   IOVs can be running concurrently. The Framework will not call
//   initializeForNewIOV or start running a new IOV unless the
//   number of active IOVs is less than that configured number.
//
//   7. The Framework guarantees that after a call is made to
//   ProductResolver::initializeForNewIOV for a particular
//   EventSetupRecordKey and iovIndex, all calls to ProductResolver::make
//   associated with that whose data is requested will be completed
//   and processing of luminosity blocks associated with that will
//   be completed before another call to ProductResolver::initializeForNewIOV
//   is made for that EventSetupRecordKey and iovIndex.

// system include files
#include <string>
#include <map>
#include <memory>
#include <set>
#include <mutex>
// user include files
#include "CondCore/CondDB/interface/ConnectionPool.h"

#include "FWCore/Framework/interface/ESProductResolverProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Concurrency/interface/SerialTaskQueue.h"

namespace edm {
  class ParameterSet;
}

namespace cond {
  class ProductResolverWrapperBase;
}

class CondDBESSource : public edm::eventsetup::ESProductResolverProvider, public edm::EventSetupRecordIntervalFinder {
public:
  using DataKey = edm::eventsetup::DataKey;
  using EventSetupRecordKey = edm::eventsetup::EventSetupRecordKey;
  typedef std::shared_ptr<cond::ProductResolverWrapperBase> ResolverP;
  typedef std::multimap<std::string, ResolverP> ResolverMap;

  typedef enum { NOREFRESH, REFRESH_ALWAYS, REFRESH_OPEN_IOVS, REFRESH_EACH_RUN, RECONNECT_EACH_RUN } RefreshPolicy;

  explicit CondDBESSource(const edm::ParameterSet&);
  ~CondDBESSource() override;

protected:
  void setIntervalFor(const EventSetupRecordKey&, const edm::IOVSyncValue&, edm::ValidityInterval&) override;

  KeyedResolversVector registerResolvers(const EventSetupRecordKey&, unsigned int iovIndex) override;

  void initConcurrentIOVs(const EventSetupRecordKey& key, unsigned int nConcurrentIOVs) override;

  bool isConcurrentFinder() const override { return true; }

private:
  // ----------member data ---------------------------

  cond::persistency::ConnectionPool m_connection;
  std::string m_connectionString;
  std::string m_frontierKey;

  // Container of ProductResolver, implemented as multi-map keyed by records
  ResolverMap m_resolvers;

  typedef std::map<std::string, cond::GTEntry_t> TagCollection;
  // the collections of tag, record/label used in this ESSource
  TagCollection m_tagCollection;
  std::map<std::string, std::pair<cond::Time_t, bool> > m_refreshTimeForRecord;
  std::map<std::string, std::pair<cond::persistency::Session, std::string> > m_sessionPool;
  std::map<std::string, std::pair<cond::persistency::Session, std::string> > m_sessionPoolForLumiConditions;
  std::map<std::string, unsigned int> m_lastRecordRuns;

  edm::SerialTaskQueue m_queue;
  std::mutex m_mutex;

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
