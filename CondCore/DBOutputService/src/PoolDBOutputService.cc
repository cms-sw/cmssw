#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/DBOutputService/interface/Exception.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "CondCore/CondDB/interface/Exception.h"
//
#include <vector>
#include <memory>
#include <cassert>

const std::string cond::service::PoolDBOutputService::kSharedResource = "PoolDBOutputService";

//In order to make PoolDBOutputService::currentTime() to work we have to keep track
// of which stream is presently being processed on a given thread during the call of
// a module which calls that method.
static thread_local int s_streamIndex = -1;

void cond::service::PoolDBOutputService::fillRecord(edm::ParameterSet& recordPset, const std::string& gTimeTypeStr) {
  Record thisrecord;

  thisrecord.m_idName = recordPset.getParameter<std::string>("record");
  thisrecord.m_tag = recordPset.getParameter<std::string>("tag");

  thisrecord.m_timetype =
      cond::time::timeTypeFromName(recordPset.getUntrackedParameter<std::string>("timetype", gTimeTypeStr));

  thisrecord.m_onlyAppendUpdatePolicy = recordPset.getUntrackedParameter<bool>("onlyAppendUpdatePolicy", false);

  thisrecord.m_refreshTime = recordPset.getUntrackedParameter<unsigned int>("refreshTime", 1);

  m_records.insert(std::make_pair(thisrecord.m_idName, thisrecord));

  cond::UserLogInfo userloginfo;
  m_logheaders.insert(std::make_pair(thisrecord.m_idName, userloginfo));
}

cond::service::PoolDBOutputService::PoolDBOutputService(const edm::ParameterSet& iConfig, edm::ActivityRegistry& iAR)
    : m_logger(iConfig.getUntrackedParameter<std::string>("jobName", "DBOutputService")),
      m_currentTimes{},
      m_session(),
      m_transactionActive(false),
      m_dbInitialised(false),
      m_records(),
      m_logheaders() {
  std::string timetypestr = iConfig.getUntrackedParameter<std::string>("timetype", "runnumber");
  m_timetype = cond::time::timeTypeFromName(timetypestr);
  m_autoCommit = iConfig.getUntrackedParameter<bool>("autoCommit", true);
  m_writeTransactionDelay = iConfig.getUntrackedParameter<unsigned int>("writeTransactionDelay", 0);
  edm::ParameterSet connectionPset = iConfig.getParameter<edm::ParameterSet>("DBParameters");
  m_connection.setParameters(connectionPset);
  m_connection.setLogDestination(m_logger);
  m_connection.configure();
  std::string connectionString = iConfig.getParameter<std::string>("connect");
  m_session = m_connection.createSession(connectionString, true);
  bool saveLogsOnDb = iConfig.getUntrackedParameter<bool>("saveLogsOnDB", false);
  if (saveLogsOnDb)
    m_logger.setDbDestination(connectionString);
  // implicit start
  //doStartTransaction();
  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters toPut = iConfig.getParameter<Parameters>("toPut");
  for (Parameters::iterator itToPut = toPut.begin(); itToPut != toPut.end(); ++itToPut)
    fillRecord(*itToPut, timetypestr);

  iAR.watchPostEndJob(this, &cond::service::PoolDBOutputService::postEndJob);
  iAR.watchPreallocate(
      [this](edm::service::SystemBounds const& iBounds) { m_currentTimes.resize(iBounds.maxNumberOfStreams()); });
  if (m_timetype == cond::timestamp) {  //timestamp
    iAR.watchPreEvent(this, &cond::service::PoolDBOutputService::preEventProcessing);
    iAR.watchPreModuleEvent(this, &cond::service::PoolDBOutputService::preModuleEvent);
    iAR.watchPostModuleEvent(this, &cond::service::PoolDBOutputService::postModuleEvent);
  } else if (m_timetype == cond::runnumber) {  //runnumber
    //NOTE: this assumes only one run is being processed at a time.
    // This is true for 7_1_X but plan are to allow multiple in flight at a time
    s_streamIndex = 0;
    iAR.watchPreGlobalBeginRun(this, &cond::service::PoolDBOutputService::preGlobalBeginRun);
  } else if (m_timetype == cond::lumiid) {
    //NOTE: this assumes only one lumi is being processed at a time.
    // This is true for 7_1_X but plan are to allow multiple in flight at a time
    s_streamIndex = 0;
    iAR.watchPreGlobalBeginLumi(this, &cond::service::PoolDBOutputService::preGlobalBeginLumi);
  }
}

cond::persistency::Session cond::service::PoolDBOutputService::newReadOnlySession(const std::string& connectionString,
                                                                                  const std::string& transactionId) {
  cond::persistency::Session ret;
  ret = m_connection.createReadOnlySession(connectionString, transactionId);
  return ret;
}

cond::persistency::Session cond::service::PoolDBOutputService::session() const { return m_session; }

void cond::service::PoolDBOutputService::lockRecords() {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  doStartTransaction();
  cond::persistency::TransactionScope scope(m_session.transaction());
  this->initDB();
  for (auto& iR : m_records) {
    if (iR.second.m_isNewTag == false) {
      cond::persistency::IOVEditor editor = m_session.editIov(iR.second.m_tag);
      editor.lock();
    }
  }
  if (m_autoCommit) {
    doCommitTransaction();
  }
  scope.close();
}

void cond::service::PoolDBOutputService::releaseLocks() {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  doStartTransaction();
  cond::persistency::TransactionScope scope(m_session.transaction());
  this->initDB();
  for (auto& iR : m_records) {
    if (iR.second.m_isNewTag == false) {
      cond::persistency::IOVEditor editor = m_session.editIov(iR.second.m_tag);
      editor.unlock();
    }
  }
  if (m_autoCommit) {
    doCommitTransaction();
  }
  scope.close();
}

std::string cond::service::PoolDBOutputService::tag(const std::string& recordName) {
  return this->lookUpRecord(recordName).m_tag;
}

bool cond::service::PoolDBOutputService::isNewTagRequest(const std::string& recordName) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  bool doCommit = false;
  if (!m_transactionActive) {
    m_session.transaction().start(true);
    doCommit = true;
  }
  bool dbexists = false;
  try {
    dbexists = initDB(true);
  } catch (const std::exception& er) {
    cond::throwException(std::string(er.what()), "PoolDBOutputService::isNewTagRequest");
  }
  if (doCommit)
    m_session.transaction().commit();
  if (!dbexists)
    return true;
  auto& myrecord = this->lookUpRecord(recordName);
  return myrecord.m_isNewTag;
}

void cond::service::PoolDBOutputService::doStartTransaction() {
  if (!m_transactionActive) {
    m_session.transaction().start(false);
    m_transactionActive = true;
  }
}

void cond::service::PoolDBOutputService::doCommitTransaction() {
  if (m_transactionActive) {
    if (m_writeTransactionDelay) {
      m_logger.logWarning() << "Waiting " << m_writeTransactionDelay << "s before commit the changes...";
      ::sleep(m_writeTransactionDelay);
    }
    m_session.transaction().commit();
    m_transactionActive = false;
  }
}

void cond::service::PoolDBOutputService::startTransaction() {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  doStartTransaction();
}

void cond::service::PoolDBOutputService::commitTransaction() {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  doCommitTransaction();
}

bool cond::service::PoolDBOutputService::initDB(bool readOnly) {
  if (!m_dbInitialised) {
    if (!m_session.existsDatabase()) {
      if (readOnly)
        return false;
      m_session.createDatabase();
    } else {
      for (auto& iR : m_records) {
        if (m_session.existsIov(iR.second.m_tag)) {
          iR.second.m_isNewTag = false;
        }
      }
    }
    m_dbInitialised = true;
  }
  return m_dbInitialised;
}

cond::service::PoolDBOutputService::Record& cond::service::PoolDBOutputService::getRecord(
    const std::string& recordName) {
  std::map<std::string, Record>::iterator it = m_records.find(recordName);
  if (it == m_records.end()) {
    cond::throwException("The record \"" + recordName + "\" has not been registered.",
                         "PoolDBOutputService::getRecord");
  }
  return it->second;
}

void cond::service::PoolDBOutputService::postEndJob() { commitTransaction(); }

void cond::service::PoolDBOutputService::preEventProcessing(edm::StreamContext const& iContext) {
  m_currentTimes[iContext.streamID().value()] = iContext.timestamp().value();
}

void cond::service::PoolDBOutputService::preModuleEvent(edm::StreamContext const& iContext,
                                                        edm::ModuleCallingContext const&) {
  s_streamIndex = iContext.streamID().value();
}

void cond::service::PoolDBOutputService::postModuleEvent(edm::StreamContext const& iContext,
                                                         edm::ModuleCallingContext const&) {
  s_streamIndex = -1;
}

void cond::service::PoolDBOutputService::preGlobalBeginRun(edm::GlobalContext const& iContext) {
  for (auto& time : m_currentTimes) {
    time = iContext.luminosityBlockID().run();
  }
}

void cond::service::PoolDBOutputService::preGlobalBeginLumi(edm::GlobalContext const& iContext) {
  for (auto& time : m_currentTimes) {
    time = iContext.luminosityBlockID().value();
  }
}

cond::service::PoolDBOutputService::~PoolDBOutputService() {}

void cond::service::PoolDBOutputService::forceInit() {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  doStartTransaction();
  cond::persistency::TransactionScope scope(m_session.transaction());
  try {
    initDB();
    if (m_autoCommit) {
      doCommitTransaction();
    }
  } catch (const std::exception& er) {
    cond::throwException(std::string(er.what()), "PoolDBOutputService::forceInit");
  }
  scope.close();
}

cond::Time_t cond::service::PoolDBOutputService::endOfTime() const { return timeTypeSpecs[m_timetype].endValue; }

cond::Time_t cond::service::PoolDBOutputService::beginOfTime() const { return timeTypeSpecs[m_timetype].beginValue; }

cond::Time_t cond::service::PoolDBOutputService::currentTime() const {
  assert(-1 != s_streamIndex);
  return m_currentTimes[s_streamIndex];
}

void cond::service::PoolDBOutputService::createNewIOV(const std::string& firstPayloadId,
                                                      cond::Time_t firstSinceTime,
                                                      const std::string& recordName) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  doStartTransaction();
  cond::persistency::TransactionScope scope(m_session.transaction());
  try {
    this->initDB();
    auto& myrecord = this->getRecord(recordName);
    if (!myrecord.m_isNewTag) {
      cond::throwException(myrecord.m_tag + " is not a new tag", "PoolDBOutputService::createNewIOV");
    }
    m_logger.logInfo() << "Creating new tag " << myrecord.m_tag << ", adding iov with since " << firstSinceTime
                       << " pointing to payload id " << firstPayloadId;
    cond::persistency::IOVEditor editor =
        m_session.createIovForPayload(firstPayloadId, myrecord.m_tag, myrecord.m_timetype, cond::SYNCH_ANY);
    editor.setDescription("New Tag");
    editor.insert(firstSinceTime, firstPayloadId);
    cond::UserLogInfo a = this->lookUpUserLogInfo(myrecord.m_idName);
    editor.flush(a.usertext);
    myrecord.m_isNewTag = false;
    if (m_autoCommit) {
      doCommitTransaction();
    }
  } catch (const std::exception& er) {
    cond::throwException(std::string(er.what()), "PoolDBOutputService::createNewIov");
  }
  scope.close();
}

// private method
void cond::service::PoolDBOutputService::createNewIOV(const std::string& firstPayloadId,
                                                      const std::string payloadType,
                                                      cond::Time_t firstSinceTime,
                                                      Record& myrecord) {
  m_logger.logInfo() << "Creating new tag " << myrecord.m_tag << " for payload type " << payloadType
                     << ", adding iov with since " << firstSinceTime;
  // FIX ME: synchronization type and description have to be passed as the other parameters?
  cond::persistency::IOVEditor editor =
      m_session.createIov(payloadType, myrecord.m_tag, myrecord.m_timetype, cond::SYNCH_ANY);
  editor.setDescription("New Tag");
  editor.insert(firstSinceTime, firstPayloadId);
  cond::UserLogInfo a = this->lookUpUserLogInfo(myrecord.m_idName);
  editor.flush(a.usertext);
  myrecord.m_isNewTag = false;
}

bool cond::service::PoolDBOutputService::appendSinceTime(const std::string& payloadId,
                                                         cond::Time_t time,
                                                         const std::string& recordName) {
  bool ret = false;
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  doStartTransaction();
  cond::persistency::TransactionScope scope(m_session.transaction());
  try {
    bool dbexists = this->initDB();
    if (!dbexists) {
      cond::throwException(std::string("Target database does not exist."), "PoolDBOutputService::appendSinceTime");
    }
    auto& myrecord = this->lookUpRecord(recordName);
    if (myrecord.m_isNewTag) {
      cond::throwException(std::string("Cannot append to non-existing tag ") + myrecord.m_tag,
                           "PoolDBOutputService::appendSinceTime");
    }
    ret = appendSinceTime(payloadId, time, myrecord);
    if (m_autoCommit) {
      doCommitTransaction();
    }
  } catch (const std::exception& er) {
    cond::throwException(std::string(er.what()), "PoolDBOutputService::appendSinceTime");
  }
  scope.close();
  return ret;
}

// private method
bool cond::service::PoolDBOutputService::appendSinceTime(const std::string& payloadId,
                                                         cond::Time_t time,
                                                         const Record& myrecord) {
  m_logger.logInfo() << "Updating existing tag " << myrecord.m_tag << ", adding iov with since " << time;
  try {
    cond::persistency::IOVEditor editor = m_session.editIov(myrecord.m_tag);
    editor.insert(time, payloadId);
    cond::UserLogInfo a = this->lookUpUserLogInfo(myrecord.m_idName);
    editor.flush(a.usertext);
  } catch (const std::exception& er) {
    cond::throwException(std::string(er.what()), "PoolDBOutputService::appendSinceTime");
  }
  return true;
}

void cond::service::PoolDBOutputService::eraseSinceTime(const std::string& payloadId,
                                                        cond::Time_t sinceTime,
                                                        const std::string& recordName) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  doStartTransaction();
  cond::persistency::TransactionScope scope(m_session.transaction());
  try {
    bool dbexists = this->initDB();
    if (!dbexists) {
      cond::throwException(std::string("Target database does not exist."), "PoolDBOutputService::eraseSinceTime");
    }
    auto& myrecord = this->lookUpRecord(recordName);
    if (myrecord.m_isNewTag) {
      cond::throwException(std::string("Cannot delete from non-existing tag ") + myrecord.m_tag,
                           "PoolDBOutputService::appendSinceTime");
    }
    m_logger.logInfo() << "Updating existing tag " << myrecord.m_tag << ", removing iov with since " << sinceTime
                       << " pointing to payload id " << payloadId;
    cond::persistency::IOVEditor editor = m_session.editIov(myrecord.m_tag);
    editor.erase(sinceTime, payloadId);
    cond::UserLogInfo a = this->lookUpUserLogInfo(recordName);
    editor.flush(a.usertext);
    if (m_autoCommit) {
      doCommitTransaction();
    }
  } catch (const std::exception& er) {
    cond::throwException(std::string(er.what()), "PoolDBOutputService::eraseSinceTime");
  }
  scope.close();
}

const cond::service::PoolDBOutputService::Record& cond::service::PoolDBOutputService::lookUpRecord(
    const std::string& recordName) {
  std::map<std::string, Record>::const_iterator it = m_records.find(recordName);
  if (it == m_records.end()) {
    cond::throwException("The record \"" + recordName + "\" has not been registered.",
                         "PoolDBOutputService::lookUpRecord");
  }
  return it->second;
}

cond::UserLogInfo& cond::service::PoolDBOutputService::lookUpUserLogInfo(const std::string& recordName) {
  std::map<std::string, cond::UserLogInfo>::iterator it = m_logheaders.find(recordName);
  if (it == m_logheaders.end())
    throw cond::Exception("Log db was not set for record " + recordName +
                          " from PoolDBOutputService::lookUpUserLogInfo");
  return it->second;
}

void cond::service::PoolDBOutputService::closeIOV(Time_t lastTill, const std::string& recordName) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  doStartTransaction();
  cond::persistency::TransactionScope scope(m_session.transaction());
  try {
    bool dbexists = this->initDB();
    if (!dbexists) {
      cond::throwException(std::string("Target database does not exist."), "PoolDBOutputService::closeIOV");
    }
    auto& myrecord = lookUpRecord(recordName);
    if (myrecord.m_isNewTag) {
      cond::throwException(std::string("Cannot close non-existing tag ") + myrecord.m_tag,
                           "PoolDBOutputService::closeIOV");
    }
    m_logger.logInfo() << "Updating existing tag " << myrecord.m_tag << ", closing with end of validity " << lastTill;
    cond::persistency::IOVEditor editor = m_session.editIov(myrecord.m_tag);
    editor.setEndOfValidity(lastTill);
    editor.flush("Tag closed.");
    if (m_autoCommit) {
      doCommitTransaction();
    }
  } catch (const std::exception& er) {
    cond::throwException(std::string(er.what()), "PoolDBOutputService::closeIOV");
  }
  scope.close();
}

void cond::service::PoolDBOutputService::setLogHeaderForRecord(const std::string& recordName,
                                                               const std::string& dataprovenance,
                                                               const std::string& usertext) {
  cond::UserLogInfo& myloginfo = this->lookUpUserLogInfo(recordName);
  myloginfo.provenance = dataprovenance;
  myloginfo.usertext = usertext;
}

// Still required.
bool cond::service::PoolDBOutputService::getTagInfo(const std::string& recordName, cond::TagInfo_t& result) {
  auto& record = lookUpRecord(recordName);
  result.name = record.m_tag;
  m_logger.logDebug() << "Fetching tag info for " << record.m_tag;
  bool ret = false;
  //use iovproxy to find out.
  if (m_session.existsIov(record.m_tag)) {
    cond::persistency::IOVProxy iov = m_session.readIov(record.m_tag);
    result.lastInterval = iov.getLast();
    ret = true;
  }
  return ret;
}

// Still required.
bool cond::service::PoolDBOutputService::tagInfo(const std::string& recordName, cond::TagInfo_t& result) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  bool ret = false;
  bool doCommit = false;
  if (!m_transactionActive) {
    m_session.transaction().start(true);
    doCommit = true;
  }
  bool dbexists = false;
  cond::persistency::TransactionScope scope(m_session.transaction());
  try {
    dbexists = initDB(true);
    if (dbexists) {
      ret = getTagInfo(recordName, result);
    }
  } catch (const std::exception& er) {
    cond::throwException(std::string(er.what()), "PoolDBOutputService::tagInfo");
  }
  if (doCommit)
    m_session.transaction().commit();
  scope.close();
  return ret;
}
