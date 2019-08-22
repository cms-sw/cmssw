#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/CaliGeneralDat.h"
#include "OnlineDB/EcalCondDB/interface/CaliTag.h"
#include "OnlineDB/EcalCondDB/interface/CaliIOV.h"

using namespace std;
using namespace oracle::occi;

CaliGeneralDat::CaliGeneralDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;

  m_numEvents = 0;
  m_comments = "none";
}

CaliGeneralDat::~CaliGeneralDat() {}

void CaliGeneralDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO cali_general_dat (iov_id, logic_id, "
        "num_events, comments) "
        "VALUES (:iov_id, :logic_id, "
        ":3, :4)");
  } catch (SQLException& e) {
    throw(std::runtime_error("CaliGeneralDat::prepareWrite():  " + e.getMessage()));
  }
}

void CaliGeneralDat::writeDB(const EcalLogicID* ecid, const CaliGeneralDat* item, CaliIOV* iov) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("CaliGeneralDat::writeDB:  IOV not in DB"));
  }

  int logicID = ecid->getLogicID();
  if (!logicID) {
    throw(std::runtime_error("CaliGeneralDat::writeDB:  Bad EcalLogicID"));
  }

  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setInt(3, item->getNumEvents());
    m_writeStmt->setString(4, item->getComments());

    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("CaliGeneralDat::writeDB():  " + e.getMessage()));
  }
}

void CaliGeneralDat::fetchData(std::map<EcalLogicID, CaliGeneralDat>* fillMap, CaliIOV* iov) noexcept(false) {
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) {
    //  throw(std::runtime_error("CaliGeneralDat::writeDB:  IOV not in DB"));
    return;
  }

  try {
    m_readStmt->setSQL(
        "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
        "d.num_events, d.comments "
        "FROM channelview cv JOIN cali_general_dat d "
        "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
        "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();

    std::pair<EcalLogicID, CaliGeneralDat> p;
    CaliGeneralDat dat;
    while (rset->next()) {
      p.first = EcalLogicID(rset->getString(1),   // name
                            rset->getInt(2),      // logic_id
                            rset->getInt(3),      // id1
                            rset->getInt(4),      // id2
                            rset->getInt(5),      // id3
                            rset->getString(6));  // maps_to

      dat.setNumEvents(rset->getInt(7));
      dat.setComments(rset->getString(8));

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException& e) {
    throw(std::runtime_error("CaliGeneralDat::fetchData():  " + e.getMessage()));
  }
}
