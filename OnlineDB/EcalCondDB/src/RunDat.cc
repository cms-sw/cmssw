#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/RunDat.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

RunDat::RunDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_numEvents = 0;
}

RunDat::~RunDat() {}

void RunDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO run_dat (iov_id, logic_id, "
        "num_events) "
        "VALUES (:iov_id, :logic_id, "
        ":num_events)");
  } catch (SQLException& e) {
    throw(std::runtime_error("RunDat::prepareWrite():  " + e.getMessage()));
  }
}

void RunDat::writeDB(const EcalLogicID* ecid, const RunDat* item, RunIOV* iov) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("RunDat::writeDB:  IOV not in DB"));
  }

  int logicID = ecid->getLogicID();
  if (!logicID) {
    throw(std::runtime_error("RunDat::writeDB:  Bad EcalLogicID"));
  }

  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setInt(3, item->getNumEvents());

    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("RunDat::writeDB():  " + e.getMessage()));
  }
}

void RunDat::fetchData(map<EcalLogicID, RunDat>* fillMap, RunIOV* iov) noexcept(false) {
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) {
    //  throw(std::runtime_error("RunDat::writeDB:  IOV not in DB"));
    return;
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL(
        "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
        "d.num_events "
        "FROM channelview cv JOIN run_dat d "
        "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
        "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();

    std::pair<EcalLogicID, RunDat> p;
    RunDat dat;
    while (rset->next()) {
      p.first = EcalLogicID(rset->getString(1),   // name
                            rset->getInt(2),      // logic_id
                            rset->getInt(3),      // id1
                            rset->getInt(4),      // id2
                            rset->getInt(5),      // id3
                            rset->getString(6));  // maps_to

      dat.setNumEvents(rset->getInt(7));

      p.second = dat;
      fillMap->insert(p);
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("RunDat::fetchData():  " + e.getMessage()));
  }
}
