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
  m_autoCommit = iConfig.getUntrackedParameter<bool>("autoCommit", false);
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
  doStartTransaction();

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

std::string cond::service::PoolDBOutputService::tag(const std::string& recordName) {
  return this->lookUpRecord(recordName).m_tag;
}

bool cond::service::PoolDBOutputService::isNewTagRequest(const std::string& recordName) {
  Record& myrecord = this->lookUpRecord(recordName);
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

void cond::service::PoolDBOutputService::initDB() {
  if (!m_dbInitialised) {
    if (!m_session.existsDatabase())
      m_session.createDatabase();
    else {
      for (auto& iR : m_records) {
        if (m_session.existsIov(iR.second.m_tag))
          iR.second.m_isNewTag = false;
      }
    }
    m_dbInitialised = true;
  }
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
  Record& myrecord = this->lookUpRecord(recordName);
  if (!myrecord.m_isNewTag) {
    cond::throwException(myrecord.m_tag + " is not a new tag", "PoolDBOutputService::createNewIOV");
  }
  m_logger.logInfo() << "Creating new tag " << myrecord.m_tag << ", adding iov with since " << firstSinceTime
                     << " pointing to payload id " << firstPayloadId;
  doStartTransaction();
  cond::persistency::TransactionScope scope(m_session.transaction());
  try {
    this->initDB();
    cond::persistency::IOVEditor editor =
        m_session.createIovForPayload(firstPayloadId, myrecord.m_tag, myrecord.m_timetype, cond::SYNCH_ANY);
    editor.setDescription("New Tag");
    editor.insert(firstSinceTime, firstPayloadId);
    cond::UserLogInfo a = this->lookUpUserLogInfo(myrecord.m_idName);
    editor.flush(a.usertext);
    myrecord.m_isNewTag = false;
  } catch (const std::exception& er) {
    cond::throwException(std::string(er.what()), "PoolDBOutputService::createNewIov");
  }
  scope.close();
}

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
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  Record& myrecord = this->lookUpRecord(recordName);
  if (myrecord.m_isNewTag) {
    cond::throwException(std::string("Cannot append to non-existing tag ") + myrecord.m_tag,
                         "PoolDBOutputService::appendSinceTime");
  }
  bool ret = false;
  doStartTransaction();
  cond::persistency::TransactionScope scope(m_session.transaction());
  try {
    ret = appendSinceTime(payloadId, time, myrecord);
  } catch (const std::exception& er) {
    cond::throwException(std::string(er.what()), "PoolDBOutputService::appendSinceTime");
  }
  scope.close();
  return ret;
}

bool cond::service::PoolDBOutputService::appendSinceTime(const std::string& payloadId,
                                                         cond::Time_t time,
                                                         Record& myrecord) {
  m_logger.logInfo() << "Updating existing tag " << myrecord.m_tag << ", adding iov with since " << time;
  std::string payloadType("");
  try {
    cond::persistency::IOVEditor editor = m_session.editIov(myrecord.m_tag);
    payloadType = editor.payloadType();
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
  Record& myrecord = this->lookUpRecord(recordName);
  if (myrecord.m_isNewTag) {
    cond::throwException(std::string("Cannot delete from non-existing tag ") + myrecord.m_tag,
                         "PoolDBOutputService::appendSinceTime");
  }
  m_logger.logInfo() << "Updating existing tag " << myrecord.m_tag << ", removing iov with since " << sinceTime
                     << " pointing to payload id " << payloadId;
  doStartTransaction();
  cond::persistency::TransactionScope scope(m_session.transaction());
  try {
    cond::persistency::IOVEditor editor = m_session.editIov(myrecord.m_tag);
    editor.erase(sinceTime, payloadId);
    cond::UserLogInfo a = this->lookUpUserLogInfo(recordName);
    editor.flush(a.usertext);

  } catch (const std::exception& er) {
    cond::throwException(std::string(er.what()), "PoolDBOutputService::eraseSinceTime");
  }
  scope.close();
}

cond::service::PoolDBOutputService::Record& cond::service::PoolDBOutputService::lookUpRecord(
    const std::string& recordName) {
  std::map<std::string, Record>::iterator it = m_records.find(recordName);
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
  Record& myrecord = lookUpRecord(recordName);
  if (myrecord.m_isNewTag) {
    cond::throwException(std::string("Cannot close non-existing tag ") + myrecord.m_tag,
                         "PoolDBOutputService::closeIOV");
  }
  m_logger.logInfo() << "Updating existing tag " << myrecord.m_tag << ", closing with end of validity " << lastTill;
  doStartTransaction();
  cond::persistency::TransactionScope scope(m_session.transaction());
  try {
    cond::persistency::IOVEditor editor = m_session.editIov(myrecord.m_tag);
    editor.setEndOfValidity(lastTill);
    editor.flush("Tag closed.");
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
  Record& record = lookUpRecord(recordName);
  result.name = record.m_tag;
  m_logger.logDebug() << "Fetching tag info for " << record.m_tag;
  doStartTransaction();
  bool ret = false;
  cond::persistency::TransactionScope scope(m_session.transaction());
  try {
    //use iovproxy to find out.
    if (m_session.existsIov(record.m_tag)) {
      cond::persistency::IOVProxy iov = m_session.readIov(record.m_tag);
      result.lastInterval = iov.getLast();
      ret = true;
    }
  } catch (const std::exception& er) {
    cond::throwException(std::string(er.what()), "PoolDBOutputService::tagInfo");
  }
  scope.close();
  return ret;
}

// Still required.
bool cond::service::PoolDBOutputService::tagInfo(const std::string& recordName, cond::TagInfo_t& result) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  return getTagInfo(recordName, result);
}
