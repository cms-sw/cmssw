#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/RunConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

RunConfigDat::RunConfigDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_configTag = "none";
  m_configVer = 0;
}

RunConfigDat::~RunConfigDat() {}

void RunConfigDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO run_config_dat (iov_id, logic_id, "
        "config_tag, config_ver) "
        "VALUES (:iov_id, :logic_id, "
        ":config_tag, :config_ver)");
  } catch (SQLException& e) {
    throw(std::runtime_error("RunConfigDat::prepareWrite():  " + e.getMessage()));
  }
}

void RunConfigDat::writeDB(const EcalLogicID* ecid, const RunConfigDat* item, RunIOV* iov) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("RunConfigDat::writeDB:  IOV not in DB"));
  }

  int logicID = ecid->getLogicID();
  if (!logicID) {
    throw(std::runtime_error("RunConfigDat::writeDB:  Bad EcalLogicID"));
  }

  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setString(3, item->getConfigTag());
    m_writeStmt->setInt(4, item->getConfigVersion());

    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("RunConfigDat::writeDB():  " + e.getMessage()));
  }
}

void RunConfigDat::fetchData(map<EcalLogicID, RunConfigDat>* fillMap, RunIOV* iov) noexcept(false) {
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) {
    //  throw(std::runtime_error("RunConfigDat::writeDB:  IOV not in DB"));
    return;
  }

  try {
    m_readStmt->setSQL(
        "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
        "d.config_tag, d.config_ver "
        "FROM channelview cv JOIN run_config_dat d "
        "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
        "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();

    std::pair<EcalLogicID, RunConfigDat> p;
    RunConfigDat dat;
    while (rset->next()) {
      p.first = EcalLogicID(rset->getString(1),   // name
                            rset->getInt(2),      // logic_id
                            rset->getInt(3),      // id1
                            rset->getInt(4),      // id2
                            rset->getInt(5),      // id3
                            rset->getString(6));  // maps_to

      dat.setConfigTag(rset->getString(7));
      dat.setConfigVersion(rset->getInt(8));

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException& e) {
    throw(std::runtime_error("RunConfigDat::fetchData():  " + e.getMessage()));
  }
}
