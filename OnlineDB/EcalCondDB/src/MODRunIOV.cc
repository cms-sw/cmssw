#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/MODRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;

MODRunIOV::MODRunIOV() {
  m_conn = nullptr;
  m_ID = 0;
  m_runIOV = RunIOV();
  m_subRunNum = 0;
  m_subRunStart = Tm();
  m_subRunEnd = Tm();
}

MODRunIOV::~MODRunIOV() {}

void MODRunIOV::setID(int id) { m_ID = id; }

void MODRunIOV::setRunIOV(const RunIOV& iov) {
  if (iov != m_runIOV) {
    m_ID = 0;
    m_runIOV = iov;
  }
}

RunIOV MODRunIOV::getRunIOV() { return m_runIOV; }

void MODRunIOV::setSubRunNumber(subrun_t subrun) {
  if (subrun != m_subRunNum) {
    m_ID = 0;
    m_subRunNum = subrun;
  }
}

run_t MODRunIOV::getSubRunNumber() const { return m_subRunNum; }

void MODRunIOV::setSubRunStart(const Tm& start) {
  if (start != m_subRunStart) {
    m_ID = 0;
    m_subRunStart = start;
  }
}

Tm MODRunIOV::getSubRunStart() const { return m_subRunStart; }

void MODRunIOV::setSubRunEnd(const Tm& end) {
  if (end != m_subRunEnd) {
    m_ID = 0;
    m_subRunEnd = end;
  }
}

Tm MODRunIOV::getSubRunEnd() const { return m_subRunEnd; }

int MODRunIOV::fetchID() noexcept(false) {
  // Return from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();

  // fetch the parent IDs
  int runIOVID;
  this->fetchParentIDs(&runIOVID);

  if (!runIOVID) {
    return 0;
  }

  DateHandler dh(m_env, m_conn);

  if (m_subRunEnd.isNull()) {
    m_subRunEnd = dh.getPlusInfTm();
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL(
        "SELECT iov_id FROM OD_run_iov "
        "WHERE "
        "run_iov_id   = :1 AND "
        "subrun_num   = :2 AND "
        "subrun_start = :3 AND "
        "subrun_end   = :4");

    stmt->setInt(1, runIOVID);
    stmt->setInt(2, m_subRunNum);
    stmt->setDate(3, dh.tmToDate(m_subRunStart));
    stmt->setDate(4, dh.tmToDate(m_subRunEnd));

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("MODRunIOV::fetchID:  " + e.getMessage()));
  }

  return m_ID;
}

void MODRunIOV::setByID(int id) noexcept(false) {
  this->checkConnection();

  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("SELECT run_iov_id, subrun_num, subrun_start, subrun_end FROM OD_run_iov WHERE iov_id = :1");
    stmt->setInt(1, id);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      int runIOVID = rset->getInt(1);
      m_subRunNum = rset->getInt(2);
      Date startDate = rset->getDate(3);
      Date endDate = rset->getDate(4);

      m_subRunStart = dh.dateToTm(startDate);
      m_subRunEnd = dh.dateToTm(endDate);

      m_runIOV.setConnection(m_env, m_conn);
      m_runIOV.setByID(runIOVID);

      m_ID = id;
    } else {
      throw(std::runtime_error("MODRunIOV::setByID:  Given id is not in the database"));
    }

    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("MODRunIOV::setByID:  " + e.getMessage()));
  }
}

int MODRunIOV::writeDB() noexcept(false) {
  this->checkConnection();

  // Check if this IOV has already been written
  if (this->fetchID()) {
    return m_ID;
  }

  // fetch Parent IDs
  int runIOVID;
  this->fetchParentIDs(&runIOVID);

  // Validate the data, use infinity-till convention
  DateHandler dh(m_env, m_conn);

  if (m_subRunStart.isNull()) {
    throw(std::runtime_error("MODRunIOV::writeDB:  Must setSubRunStart before writing"));
  }

  if (m_subRunEnd.isNull()) {
    m_subRunEnd = dh.getPlusInfTm();
  }

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL(
        "INSERT INTO od_run_iov (iov_id,  run_iov_id, subrun_num, subrun_start, subrun_end) "
        "VALUES (OD_run_iov_sq.NextVal, :1, :2, :3, :4)");
    stmt->setInt(1, runIOVID);
    stmt->setInt(2, m_subRunNum);
    stmt->setDate(3, dh.tmToDate(m_subRunStart));
    stmt->setDate(4, dh.tmToDate(m_subRunEnd));

    stmt->executeUpdate();

    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("MODRunIOV::writeDB:  " + e.getMessage()));
  }

  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("MODRunIOV::writeDB:  Failed to write"));
  }

  return m_ID;
}

void MODRunIOV::fetchParentIDs(int* runIOVID) noexcept(false) {
  // get the RunIOV
  m_runIOV.setConnection(m_env, m_conn);
  *runIOVID = m_runIOV.fetchID();

  if (!*runIOVID) {
    throw(std::runtime_error("MODRunIOV:  Given RunIOV does not exist in DB"));
  }
}

void MODRunIOV::setByRun(RunIOV* runiov, subrun_t subrun) noexcept(false) {
  this->checkConnection();

  runiov->setConnection(m_env, m_conn);
  int runIOVID = runiov->fetchID();

  if (!runIOVID) {
    throw(std::runtime_error("MODRunIOV::setByRun:  Given RunIOV does not exist in DB"));
  }

  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL(
        "SELECT iov_id, subrun_start, subrun_end FROM OD_run_iov "
        "WHERE run_iov_id = :1 AND subrun_num = :2");
    stmt->setInt(1, runIOVID);
    stmt->setInt(2, subrun);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      m_runIOV = *runiov;
      m_subRunNum = subrun;

      m_ID = rset->getInt(1);
      Date startDate = rset->getDate(2);
      Date endDate = rset->getDate(3);

      m_subRunStart = dh.dateToTm(startDate);
      m_subRunEnd = dh.dateToTm(endDate);
    } else {
      throw(std::runtime_error("MODRunIOV::setByRun:  Given subrun is not in the database"));
    }

    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("MODRunIOV::setByRun:  " + e.getMessage()));
  }
}
