#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODLTCCycle.h"

using namespace std;
using namespace oracle::occi;

ODLTCCycle::ODLTCCycle() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;
  //
  m_ID = 0;
  m_ltc_config_id = 0;
}

ODLTCCycle::~ODLTCCycle() {}

void ODLTCCycle::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO ECAL_LTC_Cycle (cycle_id, ltc_configuration_id ) "
        "VALUES (:1, :2 )");
  } catch (SQLException &e) {
    throw(std::runtime_error("ODLTCCycle::prepareWrite():  " + e.getMessage()));
  }
}

void ODLTCCycle::writeDB() noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  try {
    m_writeStmt->setInt(1, this->getId());
    m_writeStmt->setInt(2, this->getLTCConfigurationID());

    m_writeStmt->executeUpdate();

  } catch (SQLException &e) {
    throw(std::runtime_error("ODLTCCycle::writeDB:  " + e.getMessage()));
  }

  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODLTCCycle::writeDB:  Failed to write"));
  }
}

void ODLTCCycle::clear() { m_ltc_config_id = 0; }

int ODLTCCycle::fetchID() noexcept(false) {
  // Return from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement *stmt = m_conn->createStatement();
    stmt->setSQL(
        "SELECT cycle_id, ltc_configuration_id FROM ecal_ltc_cycle "
        "WHERE cycle_id = :1 ");
    stmt->setInt(1, m_ID);
    ResultSet *rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
      m_ltc_config_id = rset->getInt(2);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("ODLTCCycle::fetchID:  " + e.getMessage()));
  }

  return m_ID;
}

void ODLTCCycle::setByID(int id) noexcept(false) {
  this->checkConnection();

  try {
    Statement *stmt = m_conn->createStatement();
    stmt->setSQL(
        "SELECT cycle_id, ltc_configuration_id FROM ecal_ltc_cycle "
        "WHERE cycle_id = :1 ");
    stmt->setInt(1, id);
    ResultSet *rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
      m_ltc_config_id = rset->getInt(2);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("ODLTCCycle::fetchID:  " + e.getMessage()));
  }
}

void ODLTCCycle::fetchData(ODLTCCycle *result) noexcept(false) {
  this->checkConnection();
  result->clear();

  if (result->getId() == 0) {
    throw(std::runtime_error("ODLTCConfig::fetchData(): no Id defined for this ODLTCConfig "));
  }

  try {
    m_readStmt->setSQL(
        "SELECT  ltc_configuration_id FROM ecal_ltc_cycle "
        "WHERE cycle_id = :1 ");

    m_readStmt->setInt(1, result->getId());
    ResultSet *rset = m_readStmt->executeQuery();

    rset->next();

    result->setLTCConfigurationID(rset->getInt(1));

  } catch (SQLException &e) {
    throw(std::runtime_error("ODLTCCycle::fetchData():  " + e.getMessage()));
  }
}

void ODLTCCycle::insertConfig() noexcept(false) {
  try {
    prepareWrite();
    writeDB();
    m_conn->commit();
    terminateWriteStatement();
  } catch (std::runtime_error &e) {
    m_conn->rollback();
    throw(e);
  } catch (...) {
    m_conn->rollback();
    throw(std::runtime_error("EcalCondDBInterface::insertDataSet:  Unknown exception caught"));
  }
}
