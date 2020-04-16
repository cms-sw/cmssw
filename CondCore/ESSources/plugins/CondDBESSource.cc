//
// Package:     CondCore/ESSources
// Module:      CondDBESSource
//
// Description: <one line class summary>
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Zhen Xie
// Fixes and other changes: Giacomo Govi
//
#include "CondDBESSource.h"

#include <boost/algorithm/string.hpp>
#include "CondCore/CondDB/interface/Exception.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/Types.h"
#include "CondCore/CondDB/interface/Utils.h"

#include "CondCore/ESSources/interface/ProxyFactory.h"
#include "CondCore/ESSources/interface/DataProxy.h"

#include "CondCore/CondDB/interface/PayloadProxy.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <exception>

#include <iomanip>

namespace {
  /* utility ot build the name of the plugin corresponding to a given record
     se ESSources
   */
  std::string buildName(std::string const& iRecordName) { return iRecordName + std::string("@NewProxy"); }

  std::string joinRecordAndLabel(std::string const& iRecordName, std::string const& iLabelName) {
    return iRecordName + std::string("@") + iLabelName;
  }

  /* utility class to return a IOVs associated to a given "name"
     This implementation return the IOV associated to a record...
     It is essentialy a workaround to get the full IOV out of the tag colector
     that is not accessible as hidden in the ESSource
     FIXME: need to support label??
   */
  class CondGetterFromESSource : public cond::persistency::CondGetter {
  public:
    CondGetterFromESSource(CondDBESSource::ProxyMap const& ip) : m_proxies(ip) {}
    ~CondGetterFromESSource() override {}

    cond::persistency::IOVProxy get(std::string name) const override {
      CondDBESSource::ProxyMap::const_iterator p = m_proxies.find(name);
      if (p != m_proxies.end())
        return (*p).second->iovProxy();
      return cond::persistency::IOVProxy();
    }

    CondDBESSource::ProxyMap const& m_proxies;
  };

  // This needs to be re-design and implemented...
  // dump the state of a DataProxy
  void dumpInfo(std::ostream& out, std::string const& recName, cond::DataProxyWrapperBase const& proxy) {
    //cond::SequenceState state(proxy.proxy()->iov().state());
    out << recName << " / " << proxy.label() << ": " << proxy.connString() << ", " << proxy.tag()
        << "\n  "
        //	<< state.size() << ", " << state.revision()  << ", "
        //	<< cond::time::to_boost(state.timestamp())     << "\n  "
        //	<< state.comment()
        << "\n  "
        //	<< "refresh " << proxy.proxy()->stats.nRefresh
        //	<< "/" << proxy.proxy()->stats.nArefresh
        //	<< ", reconnect " << proxy.proxy()->stats.nReconnect
        //	<< "/" << proxy.proxy()->stats.nAreconnect
        //	<< ", make " << proxy.proxy()->stats.nMake
        //	<< ", load " << proxy.proxy()->stats.nLoad
        ;
    //if ( proxy.proxy()->stats.nLoad>0) {
    out << "Time look up, payloadIds:" << std::endl;
    const auto& pids = *proxy.requests();
    for (auto id : pids)
      out << "   " << id.since << " - " << id.till << " : " << id.payloadId << std::endl;
  }

}  // namespace

/*
 *  config Param
 *  RefreshEachRun: if true will refresh the IOV at each new run (or lumiSection)
 *  DumpStat: if true dump the statistics of all DataProxy (currently on cout)
 *  DBParameters: configuration set of the connection
 *  globaltag: The GlobalTag
 *  toGet: list of record label tag connection-string to add/overwrite the content of the global-tag
 */
CondDBESSource::CondDBESSource(const edm::ParameterSet& iConfig)
    : m_connection(),
      m_connectionString(""),
      m_lastRun(0),   // for the stat
      m_lastLumi(0),  // for the stat
      m_policy(NOREFRESH),
      m_doDump(iConfig.getUntrackedParameter<bool>("DumpStat", false)) {
  if (iConfig.getUntrackedParameter<bool>("RefreshAlways", false)) {
    m_policy = REFRESH_ALWAYS;
  }
  if (iConfig.getUntrackedParameter<bool>("RefreshOpenIOVs", false)) {
    m_policy = REFRESH_OPEN_IOVS;
  }
  if (iConfig.getUntrackedParameter<bool>("RefreshEachRun", false)) {
    m_policy = REFRESH_EACH_RUN;
  }
  if (iConfig.getUntrackedParameter<bool>("ReconnectEachRun", false)) {
    m_policy = RECONNECT_EACH_RUN;
  }

  Stats s = {0, 0, 0, 0, 0, 0, 0, 0};
  m_stats = s;

  /*parameter set parsing
   */
  std::string globaltag("");
  if (iConfig.exists("globaltag")) {
    globaltag = iConfig.getParameter<std::string>("globaltag");
    // the global tag _requires_ a connection string
    m_connectionString = iConfig.getParameter<std::string>("connect");
  } else if (iConfig.exists("connect"))  // default connection string
    m_connectionString = iConfig.getParameter<std::string>("connect");

  // snapshot
  boost::posix_time::ptime snapshotTime;
  if (iConfig.exists("snapshotTime")) {
    std::string snapshotTimeString = iConfig.getParameter<std::string>("snapshotTime");
    if (!snapshotTimeString.empty())
      snapshotTime = boost::posix_time::time_from_string(snapshotTimeString);
  }

  // connection configuration
  if (iConfig.exists("DBParameters")) {
    edm::ParameterSet connectionPset = iConfig.getParameter<edm::ParameterSet>("DBParameters");
    m_connection.setParameters(connectionPset);
  }
  m_connection.configure();

  // load specific record/tag info - it will overwrite the global tag ( if any )
  std::map<std::string, cond::GTEntry_t> replacements;
  std::map<std::string, boost::posix_time::ptime> specialSnapshots;
  if (iConfig.exists("toGet")) {
    typedef std::vector<edm::ParameterSet> Parameters;
    Parameters toGet = iConfig.getParameter<Parameters>("toGet");
    for (Parameters::iterator itToGet = toGet.begin(); itToGet != toGet.end(); ++itToGet) {
      std::string recordname = itToGet->getParameter<std::string>("record");
      if (recordname.empty())
        throw cond::Exception("ESSource: The record name has not been provided in a \"toGet\" entry.");
      std::string labelname = itToGet->getUntrackedParameter<std::string>("label", "");
      std::string pfn("");
      if (m_connectionString.empty() || itToGet->exists("connect"))
        pfn = itToGet->getParameter<std::string>("connect");
      std::string tag("");
      std::string fqTag("");
      if (itToGet->exists("tag")) {
        tag = itToGet->getParameter<std::string>("tag");
        fqTag = cond::persistency::fullyQualifiedTag(tag, pfn);
      }
      boost::posix_time::ptime tagSnapshotTime =
          boost::posix_time::time_from_string(std::string(cond::time::MAX_TIMESTAMP));
      if (itToGet->exists("snapshotTime"))
        tagSnapshotTime = boost::posix_time::time_from_string(itToGet->getParameter<std::string>("snapshotTime"));
      if (itToGet->exists("refreshTime")) {
        cond::Time_t refreshTime = itToGet->getParameter<unsigned long long>("refreshTime");
        m_refreshTimeForRecord.insert(std::make_pair(recordname, refreshTime));
      }

      std::string recordLabelKey = joinRecordAndLabel(recordname, labelname);
      replacements.insert(
          std::make_pair(recordLabelKey, cond::GTEntry_t(std::make_tuple(recordname, labelname, fqTag))));
      specialSnapshots.insert(std::make_pair(recordLabelKey, tagSnapshotTime));
    }
  }

  // get the global tag, merge with "replacement" store in "tagCollection"
  std::vector<std::string> globaltagList;
  std::vector<std::string> connectList;
  std::vector<std::string> pfnPrefixList;
  std::vector<std::string> pfnPostfixList;
  if (!globaltag.empty()) {
    std::string pfnPrefix(iConfig.getUntrackedParameter<std::string>("pfnPrefix", ""));
    std::string pfnPostfix(iConfig.getUntrackedParameter<std::string>("pfnPostfix", ""));
    boost::split(globaltagList, globaltag, boost::is_any_of("|"), boost::token_compress_off);
    fillList(m_connectionString, connectList, globaltagList.size(), "connection");
    fillList(pfnPrefix, pfnPrefixList, globaltagList.size(), "pfnPrefix");
    fillList(pfnPostfix, pfnPostfixList, globaltagList.size(), "pfnPostfix");
  }

  cond::GTMetadata_t gtMetadata;
  fillTagCollectionFromDB(connectList, pfnPrefixList, pfnPostfixList, globaltagList, replacements, gtMetadata);
  // if no job specific setting has been found, use the GT timestamp
  if (snapshotTime.is_not_a_date_time())
    snapshotTime = gtMetadata.snapshotTime;

  TagCollection::iterator it;
  TagCollection::iterator itBeg = m_tagCollection.begin();
  TagCollection::iterator itEnd = m_tagCollection.end();

  std::map<std::string, cond::persistency::Session> sessions;

  /* load DataProxy Plugin (it is strongly typed due to EventSetup ideosyncrasis)
   * construct proxy
   * contrary to EventSetup the "object-name" is not used as identifier: multiple entries in a record are
   * dinstinguished only by their label...
   * done in two step: first create ProxyWrapper loading ALL required dictionaries
   * this will allow to initialize POOL in one go for each "database"
   * The real initialization of the Data-Proxies is done in the second loop 
   */
  std::vector<std::unique_ptr<cond::DataProxyWrapperBase>> proxyWrappers(m_tagCollection.size());
  size_t ipb = 0;
  for (it = itBeg; it != itEnd; ++it) {
    proxyWrappers[ipb++] = std::unique_ptr<cond::DataProxyWrapperBase>{
        cond::ProxyFactory::get()->create(buildName(it->second.recordName()))};
  }

  // now all required libraries have been loaded
  // init sessions and DataProxies
  ipb = 0;
  for (it = itBeg; it != itEnd; ++it) {
    std::string connStr = m_connectionString;
    std::string tag = it->second.tagName();
    std::pair<std::string, std::string> tagParams = cond::persistency::parseTag(it->second.tagName());
    if (!tagParams.second.empty()) {
      connStr = tagParams.second;
      tag = tagParams.first;
    }
    std::map<std::string, cond::persistency::Session>::iterator p = sessions.find(connStr);
    cond::persistency::Session nsess;
    if (p == sessions.end()) {
      std::string oracleConnStr = cond::persistency::convertoToOracleConnection(connStr);
      std::tuple<std::string, std::string, std::string> connPars =
          cond::persistency::parseConnectionString(oracleConnStr);
      std::string dbService = std::get<1>(connPars);
      std::string dbAccount = std::get<2>(connPars);
      if ((dbService == "cms_orcon_prod" || dbService == "cms_orcon_adg") && dbAccount != "CMS_CONDITIONS")
        edm::LogWarning("CondDBESSource")
            << "[WARNING] You are reading tag \"" << tag << "\" from V1 account \"" << connStr
            << "\". The concerned Conditions might be out of date." << std::endl;
      //open db get tag info (i.e. the IOV token...)
      nsess = m_connection.createReadOnlySession(connStr, "");
      sessions.insert(std::make_pair(connStr, nsess));
    } else
      nsess = (*p).second;

    // ownership...
    ProxyP proxy(std::move(proxyWrappers[ipb++]));
    //  instert in the map
    m_proxies.insert(std::make_pair(it->second.recordName(), proxy));
    // initialize
    boost::posix_time::ptime tagSnapshotTime = snapshotTime;
    auto tagSnapshotIter = specialSnapshots.find(it->first);
    if (tagSnapshotIter != specialSnapshots.end())
      tagSnapshotTime = tagSnapshotIter->second;
    // finally, if the snapshot is set to infinity, reset the snapshot to null, to take the full iov set...
    if (tagSnapshotTime == boost::posix_time::time_from_string(std::string(cond::time::MAX_TIMESTAMP)))
      tagSnapshotTime = boost::posix_time::ptime();

    proxy->lateInit(nsess, tag, tagSnapshotTime, it->second.recordLabel(), connStr);
  }

  // one loaded expose all other tags to the Proxy!
  CondGetterFromESSource visitor(m_proxies);
  ProxyMap::iterator b = m_proxies.begin();
  ProxyMap::iterator e = m_proxies.end();
  for (; b != e; b++) {
    (*b).second->proxy(0)->loadMore(visitor);

    /// required by eventsetup
    EventSetupRecordKey recordKey(EventSetupRecordKey::TypeTag::findType((*b).first));
    if (recordKey.type() != EventSetupRecordKey::TypeTag()) {
      findingRecordWithKey(recordKey);
      usingRecordWithKey(recordKey);
    }
  }

  m_stats.nData = m_proxies.size();
}

void CondDBESSource::fillList(const std::string& stringList,
                              std::vector<std::string>& listToFill,
                              const unsigned int listSize,
                              const std::string& type) {
  boost::split(listToFill, stringList, boost::is_any_of("|"), boost::token_compress_off);
  // If it is one clone it for each GT
  if (listToFill.size() == 1) {
    for (unsigned int i = 1; i < listSize; ++i) {
      listToFill.push_back(stringList);
    }
  }
  // else if they don't match the number of GTs throw an exception
  else if (listSize != listToFill.size()) {
    throw cond::Exception(
        std::string("ESSource: number of global tag components does not match number of " + type + " strings"));
  }
}

CondDBESSource::~CondDBESSource() {
  //dump info FIXME: find a more suitable place...
  if (m_doDump) {
    std::cout << "CondDBESSource Statistics" << std::endl
              << "DataProxy " << m_stats.nData << " setInterval " << m_stats.nSet << " Runs " << m_stats.nRun
              << " Lumis " << m_stats.nLumi << " Refresh " << m_stats.nRefresh << " Actual Refresh "
              << m_stats.nActualRefresh << " Reconnect " << m_stats.nReconnect << " Actual Reconnect "
              << m_stats.nActualReconnect;
    std::cout << std::endl;

    ProxyMap::iterator b = m_proxies.begin();
    ProxyMap::iterator e = m_proxies.end();
    for (; b != e; b++) {
      dumpInfo(std::cout, (*b).first, *(*b).second);
      std::cout << "\n" << std::endl;
    }

    // FIXME
    // We shall eventually close transaction and session...
  }
}

//
// invoked by EventSetUp: for a given record return the smallest IOV for which iTime is valid
// limit to next run/lumisection of Refresh is required
//
void CondDBESSource::setIntervalFor(const EventSetupRecordKey& iKey,
                                    const edm::IOVSyncValue& iTime,
                                    edm::ValidityInterval& oInterval) {
  std::string recordname = iKey.name();

  edm::LogInfo("CondDBESSource") << "Getting data for record \"" << recordname << "\" to be consumed by "
                                 << iTime.eventID() << ", timestamp: " << iTime.time().value()
                                 << "; from CondDBESSource::setIntervalFor";

  m_stats.nSet++;
  //{
  // not really required, keep here for the time being
  if (iTime.eventID().run() != m_lastRun) {
    m_lastRun = iTime.eventID().run();
    m_stats.nRun++;
  }
  if (iTime.luminosityBlockNumber() != m_lastLumi) {
    m_lastLumi = iTime.luminosityBlockNumber();
    m_stats.nLumi++;
  }
  //}
  cond::Time_t lastTime = m_lastRun;
  cond::Time_t defaultIovSize = cond::time::MAX_VAL;
  cond::Time_t minDiffTime = 1;
  bool refreshThisRecord = false;
  if (m_policy != REFRESH_ALWAYS) {
    auto iR = m_refreshTimeForRecord.find(recordname);
    refreshThisRecord = (iR != m_refreshTimeForRecord.end());
    if (refreshThisRecord) {
      lastTime = cond::time::lumiTime(m_lastRun, m_lastLumi);
      defaultIovSize = iR->second;
      minDiffTime = defaultIovSize;
    }
  }
  bool doRefresh = false;
  if (m_policy == REFRESH_EACH_RUN || m_policy == RECONNECT_EACH_RUN || refreshThisRecord) {
    // find out the last run number for the proxy of the specified record
    std::map<std::string, cond::Time_t>::iterator iRec = m_lastRecordRuns.find(recordname);
    if (iRec != m_lastRecordRuns.end()) {
      cond::Time_t lastRecordRun = iRec->second;
      cond::Time_t diffTime = lastTime - lastRecordRun;
      if (lastRecordRun > lastTime)
        diffTime = lastRecordRun - lastTime;
      if (diffTime >= minDiffTime) {
        // a refresh is required!
        doRefresh = true;
        iRec->second = lastTime;
        edm::LogInfo("CondDBESSource") << "Preparing refresh for record \"" << recordname
                                       << "\" since there has been a transition from run/lumi " << lastRecordRun
                                       << " to run/lumi " << lastTime << "; from CondDBESSource::setIntervalFor";
      }
    } else {
      doRefresh = true;
      m_lastRecordRuns.insert(std::make_pair(recordname, lastTime));
      edm::LogInfo("CondDBESSource") << "Preparing refresh for record \"" << recordname << "\" for " << iTime.eventID()
                                     << ", timestamp: " << iTime.time().value()
                                     << "; from CondDBESSource::setIntervalFor";
    }
    if (!doRefresh)
      edm::LogInfo("CondDBESSource") << "Though enabled, refresh not needed for record \"" << recordname << "\" for "
                                     << iTime.eventID() << ", timestamp: " << iTime.time().value()
                                     << "; from CondDBESSource::setIntervalFor";
  } else if (m_policy == REFRESH_ALWAYS || m_policy == REFRESH_OPEN_IOVS) {
    doRefresh = true;
    edm::LogInfo("CondDBESSource") << "Forcing refresh for record \"" << recordname << "\" for " << iTime.eventID()
                                   << ", timestamp: " << iTime.time().value()
                                   << "; from CondDBESSource::setIntervalFor";
  }

  oInterval = edm::ValidityInterval::invalidInterval();

  // compute the smallest interval (assume all objects have the same timetype....)
  cond::ValidityInterval recordValidity(1, cond::TIMELIMIT);
  cond::TimeType timetype = cond::TimeType::invalid;
  bool userTime = true;

  //FIXME use equal_range
  ProxyMap::const_iterator pmBegin = m_proxies.lower_bound(recordname);
  ProxyMap::const_iterator pmEnd = m_proxies.upper_bound(recordname);
  if (pmBegin == pmEnd) {
    edm::LogInfo("CondDBESSource") << "No DataProxy (Pluging) found for record \"" << recordname
                                   << "\"; from CondDBESSource::setIntervalFor";
    return;
  }

  for (ProxyMap::const_iterator pmIter = pmBegin; pmIter != pmEnd; ++pmIter) {
    edm::LogInfo("CondDBESSource") << "Processing record \"" << recordname << "\" and label \""
                                   << pmIter->second->label() << "\" for " << iTime.eventID()
                                   << ", timestamp: " << iTime.time().value()
                                   << "; from CondDBESSource::setIntervalFor";

    timetype = (*pmIter).second->timeType();

    cond::Time_t abtime = cond::time::fromIOVSyncValue(iTime, timetype);
    userTime = (0 == abtime);

    if (userTime)
      return;  //  oInterval invalid to avoid that make is called...

    if (doRefresh) {
      std::string recKey = joinRecordAndLabel(recordname, pmIter->second->label());
      TagCollection::const_iterator tcIter = m_tagCollection.find(recKey);
      if (tcIter == m_tagCollection.end()) {
        edm::LogInfo("CondDBESSource") << "No Tag found for record \"" << recordname << "\" and label \""
                                       << pmIter->second->label() << "\"; from CondDBESSource::setIntervalFor";
        return;
      }

      // first reconnect if required
      if (m_policy == RECONNECT_EACH_RUN || refreshThisRecord) {
        edm::LogInfo("CondDBESSource")
            << "Checking if the session must be closed and re-opened for getting correct conditions"
            << "; from CondDBESSource::setIntervalFor";
        std::stringstream transId;
        //transId << "long" << m_lastRun;
        transId << lastTime;
        std::string connStr = m_connectionString;
        std::pair<std::string, std::string> tagParams = cond::persistency::parseTag(tcIter->second.tagName());
        if (!tagParams.second.empty())
          connStr = tagParams.second;
        std::map<std::string, std::pair<cond::persistency::Session, std::string>>* sessionPool = &m_sessionPool;
        if (refreshThisRecord) {
          sessionPool = &m_sessionPoolForLumiConditions;
        }
        auto iSess = sessionPool->find(connStr);
        bool reopen = false;
        if (iSess != sessionPool->end()) {
          if (iSess->second.second != transId.str()) {
            // the available session is open for a different run: reopen
            reopen = true;
            iSess->second.second = transId.str();
          }
        } else {
          // no available session: probably first run analysed...
          iSess =
              sessionPool->insert(std::make_pair(connStr, std::make_pair(cond::persistency::Session(), transId.str())))
                  .first;
          reopen = true;
        }
        if (reopen) {
          iSess->second.first = m_connection.createReadOnlySession(connStr, transId.str());
          edm::LogInfo("CondDBESSource") << "Re-opening the session with connection string " << connStr
                                         << " and new transaction Id " << transId.str()
                                         << "; from CondDBESSource::setIntervalFor";
        }

        edm::LogInfo("CondDBESSource") << "Reconnecting to \"" << connStr << "\" for getting new payload for record \""
                                       << recordname << "\" and label \"" << pmIter->second->label()
                                       << "\" from IOV tag \"" << tcIter->second.tagName() << "\" to be consumed by "
                                       << iTime.eventID() << ", timestamp: " << iTime.time().value()
                                       << "; from CondDBESSource::setIntervalFor";
        pmIter->second->session() = iSess->second.first;
        pmIter->second->reload();
        //if( isSizeIncreased )
        //edm::LogInfo( "CondDBESSource" ) << "After reconnecting, an increased size of the IOV sequence labeled by tag \"" << tcIter->second.tag
        //				 << "\" was found; from CondDBESSource::setIntervalFor";
        //m_stats.nActualReconnect += isSizeIncreased;
        m_stats.nReconnect++;
      } else {
        edm::LogInfo("CondDBESSource") << "Refreshing IOV sequence labeled by tag \"" << tcIter->second.tagName()
                                       << "\" for getting new payload for record \"" << recordname << "\" and label \""
                                       << pmIter->second->label() << "\" to be consumed by " << iTime.eventID()
                                       << ", timestamp: " << iTime.time().value()
                                       << "; from CondDBESSource::setIntervalFor";
        pmIter->second->reload();
        //if( isSizeIncreased )
        //  edm::LogInfo( "CondDBESSource" ) << "After refreshing, an increased size of the IOV sequence labeled by tag \"" << tcIter->second.tag
        //				   << "\" was found; from CondDBESSource::setIntervalFor";
        //m_stats.nActualRefresh += isSizeIncreased;
        m_stats.nRefresh++;
      }
    }

    /*
      // make oInterval valid For Ever
    {
     oInterval = edm::ValidityInterval(cond::toIOVSyncValue(recordValidity.first,  cond::runnumber, true), 
                                       cond::toIOVSyncValue(recordValidity.second, cond::runnumber, false));
     return;
    }    
    */

    //query the IOVSequence
    cond::ValidityInterval validity = (*pmIter).second->setIntervalFor(abtime, defaultIovSize);

    edm::LogInfo("CondDBESSource") << "Validity coming from IOV sequence for record \"" << recordname
                                   << "\" and label \"" << pmIter->second->label() << "\": (" << validity.first << ", "
                                   << validity.second << ") for time (type: " << cond::timeTypeNames(timetype) << ") "
                                   << abtime << "; from CondDBESSource::setIntervalFor";

    recordValidity.first = std::max(recordValidity.first, validity.first);
    recordValidity.second = std::min(recordValidity.second, validity.second);
  }

  if (m_policy == REFRESH_OPEN_IOVS) {
    doRefresh = (recordValidity.second == cond::timeTypeSpecs[timetype].endValue);
    edm::LogInfo("CondDBESSource") << "Validity for record \"" << recordname
                                   << "\" and the corresponding label(s) coming from Condition DB: ("
                                   << recordValidity.first << ", " << recordValidity.first
                                   << ") as the last IOV element in the IOV sequence is infinity"
                                   << "; from CondDBESSource::setIntervalFor";
  }

  // to force refresh we set end-value to the minimum such an IOV can extend to: current run or lumiblock

  if ((!userTime) && recordValidity.second != 0) {
    edm::IOVSyncValue start = cond::time::toIOVSyncValue(recordValidity.first, timetype, true);
    edm::IOVSyncValue stop = doRefresh ? cond::time::limitedIOVSyncValue(iTime, timetype)
                                       : cond::time::toIOVSyncValue(recordValidity.second, timetype, false);

    if (start == edm::IOVSyncValue::invalidIOVSyncValue() && stop != edm::IOVSyncValue::invalidIOVSyncValue()) {
      start = edm::IOVSyncValue::beginOfTime();
    }
    oInterval = edm::ValidityInterval(start, stop);
  }

  edm::LogInfo("CondDBESSource") << "Setting validity for record \"" << recordname
                                 << "\" and corresponding label(s): starting at " << oInterval.first().eventID()
                                 << ", timestamp: " << oInterval.first().time().value() << ", ending at "
                                 << oInterval.last().eventID() << ", timestamp: " << oInterval.last().time().value()
                                 << ", for " << iTime.eventID() << ", timestamp: " << iTime.time().value()
                                 << "; from CondDBESSource::setIntervalFor";
}

//required by EventSetup System
edm::eventsetup::DataProxyProvider::KeyedProxiesVector CondDBESSource::registerProxies(
    const EventSetupRecordKey& iRecordKey, unsigned int iovIndex) {
  KeyedProxiesVector keyedProxiesVector;

  std::string recordname = iRecordKey.name();

  ProxyMap::const_iterator b = m_proxies.lower_bound(recordname);
  ProxyMap::const_iterator e = m_proxies.upper_bound(recordname);
  if (b == e) {
    edm::LogInfo("CondDBESSource") << "No DataProxy (Pluging) found for record \"" << recordname
                                   << "\"; from CondDBESSource::registerProxies";
    return keyedProxiesVector;
  }

  for (ProxyMap::const_iterator p = b; p != e; ++p) {
    if (nullptr != (*p).second.get()) {
      edm::eventsetup::TypeTag type = (*p).second->type();
      DataKey key(type, edm::eventsetup::IdTags((*p).second->label().c_str()));
      keyedProxiesVector.emplace_back(key, (*p).second->edmProxy(iovIndex));
    }
  }
  return keyedProxiesVector;
}

void CondDBESSource::initConcurrentIOVs(const EventSetupRecordKey& key, unsigned int nConcurrentIOVs) {
  std::string recordname = key.name();
  ProxyMap::const_iterator b = m_proxies.lower_bound(recordname);
  ProxyMap::const_iterator e = m_proxies.upper_bound(recordname);
  for (ProxyMap::const_iterator p = b; p != e; ++p) {
    if (p->second) {
      p->second->initConcurrentIOVs(nConcurrentIOVs);
    }
  }
}

// Fills tag collection from the given globaltag
void CondDBESSource::fillTagCollectionFromGT(const std::string& connectionString,
                                             const std::string& prefix,
                                             const std::string& postfix,
                                             const std::string& roottag,
                                             std::set<cond::GTEntry_t>& tagcoll,
                                             cond::GTMetadata_t& gtMetadata) {
  if (!roottag.empty()) {
    if (connectionString.empty())
      throw cond::Exception(std::string("ESSource: requested global tag ") + roottag +
                            std::string(" but not connection string given"));
    std::tuple<std::string, std::string, std::string> connPars =
        cond::persistency::parseConnectionString(connectionString);
    if (std::get<2>(connPars) == "CMS_COND_31X_GLOBALTAG") {
      edm::LogWarning("CondDBESSource")
          << "[WARNING] You are reading Global Tag \"" << roottag
          << "\" from V1 account \"CMS_COND_31X_GLOBALTAG\". The concerned Conditions might be out of date."
          << std::endl;
    } else if (roottag.rfind("::All") != std::string::npos && std::get<2>(connPars) == "CMS_CONDITIONS") {
      edm::LogWarning("CondDBESSource") << "[WARNING] You are trying to read Global Tag \"" << roottag
                                        << "\" - postfix \"::All\" should not be used for V2." << std::endl;
    }
    cond::persistency::Session session = m_connection.createSession(connectionString);
    session.transaction().start(true);
    cond::persistency::GTProxy gtp = session.readGlobalTag(roottag, prefix, postfix);
    gtMetadata.snapshotTime = gtp.snapshotTime();
    for (const auto& gte : gtp) {
      tagcoll.insert(gte);
    }
    session.transaction().commit();
  }
}

// fills tagcollection merging with replacement
// Note: it assumem the coraldbList and roottagList have the same length. This checked in the constructor that prepares the two lists before calling this method.
void CondDBESSource::fillTagCollectionFromDB(const std::vector<std::string>& connectionStringList,
                                             const std::vector<std::string>& prefixList,
                                             const std::vector<std::string>& postfixList,
                                             const std::vector<std::string>& roottagList,
                                             std::map<std::string, cond::GTEntry_t>& replacement,
                                             cond::GTMetadata_t& gtMetadata) {
  std::set<cond::GTEntry_t> tagcoll;

  auto connectionString = connectionStringList.begin();
  auto prefix = prefixList.begin();
  auto postfix = postfixList.begin();
  for (auto roottag = roottagList.begin(); roottag != roottagList.end();
       ++roottag, ++connectionString, ++prefix, ++postfix) {
    fillTagCollectionFromGT(*connectionString, *prefix, *postfix, *roottag, tagcoll, gtMetadata);
  }

  std::set<cond::GTEntry_t>::iterator tagCollIter;
  std::set<cond::GTEntry_t>::iterator tagCollBegin = tagcoll.begin();
  std::set<cond::GTEntry_t>::iterator tagCollEnd = tagcoll.end();

  // FIXME the logic is a bit perverse: can be surely linearized (at least simplified!) ....
  for (tagCollIter = tagCollBegin; tagCollIter != tagCollEnd; ++tagCollIter) {
    std::string recordLabelKey = joinRecordAndLabel(tagCollIter->recordName(), tagCollIter->recordLabel());
    std::map<std::string, cond::GTEntry_t>::iterator fid = replacement.find(recordLabelKey);
    if (fid != replacement.end()) {
      if (!fid->second.tagName().empty()) {
        cond::GTEntry_t tagMetadata(
            std::make_tuple(tagCollIter->recordName(), tagCollIter->recordLabel(), fid->second.tagName()));
        m_tagCollection.insert(std::make_pair(recordLabelKey, tagMetadata));
        edm::LogInfo("CondDBESSource") << "Replacing tag \"" << tagCollIter->tagName() << "\" for record \""
                                       << tagMetadata.recordName() << "\" and label \"" << tagMetadata.recordLabel()
                                       << "\" with tag " << tagMetadata.tagName()
                                       << "\"; from CondDBESSource::fillTagCollectionFromDB";
      } else {
        m_tagCollection.insert(std::make_pair(recordLabelKey, *tagCollIter));
      }
      replacement.erase(fid);
    } else {
      m_tagCollection.insert(std::make_pair(recordLabelKey, *tagCollIter));
    }
  }
  std::map<std::string, cond::GTEntry_t>::iterator replacementIter;
  std::map<std::string, cond::GTEntry_t>::iterator replacementBegin = replacement.begin();
  std::map<std::string, cond::GTEntry_t>::iterator replacementEnd = replacement.end();
  for (replacementIter = replacementBegin; replacementIter != replacementEnd; ++replacementIter) {
    if (replacementIter->second.tagName().empty()) {
      std::stringstream msg;
      msg << "ESSource: no tag provided for record " << replacementIter->second.recordName();
      if (!replacementIter->second.recordLabel().empty())
        msg << " and label " << replacementIter->second.recordLabel();
      throw cond::Exception(msg.str());
    }
    m_tagCollection.insert(*replacementIter);
  }
}

// backward compatibility for configuration files
class PoolDBESSource : public CondDBESSource {
public:
  explicit PoolDBESSource(const edm::ParameterSet& ps) : CondDBESSource(ps) {}
};

#include "FWCore/Framework/interface/SourceFactory.h"
//define this as a plug-in
DEFINE_FWK_EVENTSETUP_SOURCE(PoolDBESSource);
