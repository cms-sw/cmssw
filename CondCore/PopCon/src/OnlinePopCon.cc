#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/PopCon/interface/OnlinePopCon.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//#include <iostream>

namespace popcon {

  constexpr const char* const OnlinePopCon::s_version;

  OnlinePopCon::OnlinePopCon(const edm::ParameterSet& pset)
      : m_targetSession(),
        m_targetConnectionString(pset.getUntrackedParameter<std::string>("targetDBConnectionString", "")),
        m_authPath(pset.getUntrackedParameter<std::string>("authenticationPath", "")),
        m_authSys(pset.getUntrackedParameter<int>("authenticationSystem", 1)),
        m_recordName(pset.getParameter<std::string>("record")),
        m_useLockRecors(pset.getUntrackedParameter<bool>("useLockRecords", false)) {
    edm::LogInfo("OnlinePopCon")
        << "This is OnlinePopCon (Populator of Condition) v" << s_version << ".\n"
        << "Please report any problem and feature request through the JIRA project CMSCONDDB.\n";
  }

  OnlinePopCon::~OnlinePopCon() {
    if (!m_targetConnectionString.empty()) {
      m_targetSession.transaction().commit();
    }
  }

  cond::persistency::Session OnlinePopCon::preparePopCon() {
    // Initialization almost identical to PopCon
    const std::string& connectionStr = m_dbService->session().connectionString();
    m_dbService->forceInit();
    std::string tagName = m_dbService->tag(m_recordName);
    m_tagInfo.name = tagName;
    if (m_targetConnectionString.empty()) {
      m_targetSession = m_dbService->session();
      m_dbService->startTransaction();
    } else {
      cond::persistency::ConnectionPool connPool;
      connPool.setAuthenticationPath(m_authPath);
      connPool.setAuthenticationSystem(m_authSys);
      connPool.configure();
      m_targetSession = connPool.createSession(m_targetConnectionString);
      m_targetSession.transaction().start();
    }

    m_dbService->logger().logInfo() << "OnlinePopCon::preparePopCon";
    m_dbService->logger().logInfo() << "  destination DB: " << connectionStr;
    m_dbService->logger().logInfo() << "  target DB: "
                                    << (m_targetConnectionString.empty() ? connectionStr : m_targetConnectionString);

    if (m_targetSession.existsDatabase() && m_targetSession.existsIov(tagName)) {
      cond::persistency::IOVProxy iov = m_targetSession.readIov(tagName);
      m_tagInfo.size = iov.sequenceSize();
      if (m_tagInfo.size > 0) {
        m_tagInfo.lastInterval = iov.getLast();
      }
      m_dbService->logger().logInfo() << "  TAG: " << tagName << ", last since/till: " << m_tagInfo.lastInterval.since
                                      << "/" << m_tagInfo.lastInterval.till;
      m_dbService->logger().logInfo() << "  size: " << m_tagInfo.size;
    } else {
      m_dbService->logger().logInfo() << "  TAG: " << tagName << "; First writer to this new tag.";
    }
    return m_targetSession;
  }

  cond::persistency::Session OnlinePopCon::initialize() {
    // Check if DB service is available
    if (!m_dbService.isAvailable()) {
      throw Exception("OnlinePopCon", "[initialize] DBService not available");
    }

    // Start DB logging service
    m_dbLoggerReturn_ = 0;
    m_dbService->logger().start();
    m_dbService->logger().logInfo() << "OnlinePopCon::initialize - begin logging for record: " << m_recordName;

    // If requested, lock records
    if (m_useLockRecors) {
      m_dbService->logger().logInfo() << "OnlinePopCon::initialize - locking records";
      m_dbService->lockRecords();
    }

    // Prepare the rest of PopCon infrastructure
    auto session = preparePopCon();
    return session;
  }

  void OnlinePopCon::finalize() {
    // Check if DB service is available
    if (!m_dbService.isAvailable()) {
      throw Exception("OnlinePopCon", "[finalize] DBService not available");
    }

    // Release locks if previously locked
    if (m_useLockRecors) {
      m_dbService->logger().logInfo() << "OnlinePopCon::finalize - releasing locks";
      m_dbService->releaseLocks();
    }

    // Finalize PopCon infrastructure
    if (m_targetConnectionString.empty()) {
      m_dbService->commitTransaction();
    } else {
      m_targetSession.transaction().commit();
    }

    // Stop DB logging service
    m_dbService->logger().logInfo() << "OnlinePopCon::finalize - end logging for record: " << m_recordName;
    m_dbService->logger().end(m_dbLoggerReturn_);
  }

}  // namespace popcon
