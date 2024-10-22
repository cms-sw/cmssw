#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODTTCciCycle.h"

using namespace std;
using namespace oracle::occi;

ODTTCciCycle::ODTTCciCycle() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;
  //
  m_ID = 0;
  m_ttcci_config_id = 0;
}

ODTTCciCycle::~ODTTCciCycle() {}

void ODTTCciCycle::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO ECAL_TTCci_Cycle (cycle_id, ttcci_configuration_id ) "
        "VALUES (:1, :2 )");
  } catch (SQLException &e) {
    throw(std::runtime_error("ODTTCciCycle::prepareWrite():  " + e.getMessage()));
  }
}

void ODTTCciCycle::writeDB() noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  try {
    m_writeStmt->setInt(1, this->getId());
    m_writeStmt->setInt(2, this->getTTCciConfigurationID());

    m_writeStmt->executeUpdate();

  } catch (SQLException &e) {
    throw(std::runtime_error("ODTTCciCycle::writeDB:  " + e.getMessage()));
  }

  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODTTCciCycle::writeDB:  Failed to write"));
  }
}

void ODTTCciCycle::clear() { m_ttcci_config_id = 0; }

int ODTTCciCycle::fetchID() noexcept(false) {
  // Return from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement *stmt = m_conn->createStatement();
    stmt->setSQL(
        "SELECT cycle_id, ttcci_configuration_id FROM ecal_ttcci_cycle "
        "WHERE cycle_id = :1 ");
    stmt->setInt(1, m_ID);
    ResultSet *rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
      m_ttcci_config_id = rset->getInt(2);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("ODTTCciCycle::fetchID:  " + e.getMessage()));
  }

  return m_ID;
}

void ODTTCciCycle::setByID(int id) noexcept(false) {
  this->checkConnection();

  try {
    Statement *stmt = m_conn->createStatement();
    stmt->setSQL(
        "SELECT cycle_id, ttcci_configuration_id FROM ecal_ttcci_cycle "
        "WHERE cycle_id = :1 ");
    stmt->setInt(1, id);
    ResultSet *rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
      m_ttcci_config_id = rset->getInt(2);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("ODTTCciCycle::fetchID:  " + e.getMessage()));
  }
}

void ODTTCciCycle::fetchData(ODTTCciCycle *result) noexcept(false) {
  this->checkConnection();
  result->clear();

  if (result->getId() == 0) {
    throw(std::runtime_error("ODTTCciConfig::fetchData(): no Id defined for this ODTTCciConfig "));
  }

  try {
    m_readStmt->setSQL(
        "SELECT  ttcci_configuration_id FROM ecal_ttcci_cycle "
        "WHERE cycle_id = :1 ");

    m_readStmt->setInt(1, result->getId());
    ResultSet *rset = m_readStmt->executeQuery();

    rset->next();

    result->setTTCciConfigurationID(rset->getInt(1));

  } catch (SQLException &e) {
    throw(std::runtime_error("ODTTCciCycle::fetchData():  " + e.getMessage()));
  }
}

void ODTTCciCycle::insertConfig() noexcept(false) {
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
