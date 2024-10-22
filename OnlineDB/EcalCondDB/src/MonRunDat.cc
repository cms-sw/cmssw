#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/MonRunDat.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonRunOutcomeDef.h"

using namespace std;
using namespace oracle::occi;

MonRunDat::MonRunDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_numEvents = 0;
  m_outcomeDef = MonRunOutcomeDef();
  m_rootfileName = "";
  m_taskList = 0;
  m_taskOutcome = 0;
}

MonRunDat::~MonRunDat() {}

void MonRunDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO mon_run_dat (iov_id, logic_id, "
        "num_events, run_outcome_id, rootfile_name, task_list, task_outcome) "
        "VALUES (:iov_id, :logic_id, "
        ":num_events, :run_outcome_id, :rootfile_name, :task_list, :task_outcome) ");
  } catch (SQLException& e) {
    throw(std::runtime_error("MonRunDat::prepareWrite():  " + e.getMessage()));
  }
}

void MonRunDat::writeDB(const EcalLogicID* ecid, const MonRunDat* item, MonRunIOV* iov) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("MonRunDat::writeDB:  IOV not in DB"));
  }

  MonRunOutcomeDef monRunOutcomeDef = item->getMonRunOutcomeDef();  // XXX object copy every row!
  monRunOutcomeDef.setConnection(m_env, m_conn);
  int outcomeID = monRunOutcomeDef.fetchID();
  if (!outcomeID) {
    throw(std::runtime_error("MonRunDat::writeDB:  Outcome Definition not in DB"));
  }

  int logicID = ecid->getLogicID();
  if (!logicID) {
    throw(std::runtime_error("MonRunDat::writeDB:  Bad EcalLogicID"));
  }

  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setInt(3, item->getNumEvents());
    m_writeStmt->setInt(4, outcomeID);
    m_writeStmt->setString(5, item->getRootfileName());
    m_writeStmt->setInt(6, item->getTaskList());
    m_writeStmt->setInt(7, item->getTaskOutcome());

    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("MonRunDat::writeDB():  " + e.getMessage()));
  }
}

void MonRunDat::fetchData(map<EcalLogicID, MonRunDat>* fillMap, MonRunIOV* iov) noexcept(false) {
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) {
    //  throw(std::runtime_error("MonRunDat::writeDB:  IOV not in DB"));
    return;
  }

  try {
    m_readStmt->setSQL(
        "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
        "d.num_events, d.run_outcome_id, d.rootfile_name, d.task_list, d.task_outcome "
        "FROM channelview cv JOIN mon_run_dat d "
        "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
        "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();

    std::pair<EcalLogicID, MonRunDat> p;
    MonRunDat dat;
    MonRunOutcomeDef outcomeDef;
    outcomeDef.setConnection(m_env, m_conn);
    while (rset->next()) {
      p.first = EcalLogicID(rset->getString(1),   // name
                            rset->getInt(2),      // logic_id
                            rset->getInt(3),      // id1
                            rset->getInt(4),      // id2
                            rset->getInt(5),      // id3
                            rset->getString(6));  // maps_to

      dat.setNumEvents(rset->getInt(7));
      outcomeDef.setByID(rset->getInt(8));
      dat.setMonRunOutcomeDef(outcomeDef);
      dat.setRootfileName(rset->getString(9));
      dat.setTaskList(rset->getInt(10));
      dat.setTaskOutcome(rset->getInt(11));

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException& e) {
    throw(std::runtime_error("MonRunDat::fetchData():  " + e.getMessage()));
  }
}
