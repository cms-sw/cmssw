#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODCCSCycle.h"

using namespace std;
using namespace oracle::occi;

ODCCSCycle::ODCCSCycle() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;
  //
  m_ID = 0;
  m_ccs_config_id = 0;
}

ODCCSCycle::~ODCCSCycle() {}

void ODCCSCycle::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO ECAL_CCS_Cycle (cycle_id, ccs_configuration_id ) "
        "VALUES (:1, :2 )");
  } catch (SQLException &e) {
    throw(std::runtime_error("ODCCSCycle::prepareWrite():  " + e.getMessage()));
  }
}

void ODCCSCycle::writeDB() noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  try {
    m_writeStmt->setInt(1, this->getId());
    m_writeStmt->setInt(2, this->getCCSConfigurationID());

    m_writeStmt->executeUpdate();

  } catch (SQLException &e) {
    throw(std::runtime_error("ODCCSCycle::writeDB:  " + e.getMessage()));
  }

  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODCCSCycle::writeDB:  Failed to write"));
  }
}

void ODCCSCycle::clear() { m_ccs_config_id = 0; }

int ODCCSCycle::fetchID() noexcept(false) {
  // Return from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement *stmt = m_conn->createStatement();
    stmt->setSQL(
        "SELECT cycle_id, ccs_configuration_id FROM ecal_ccs_cycle "
        "WHERE cycle_id = :1 ");
    stmt->setInt(1, m_ID);
    ResultSet *rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
      m_ccs_config_id = rset->getInt(2);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("ODCCSCycle::fetchID:  " + e.getMessage()));
  }

  return m_ID;
}

void ODCCSCycle::setByID(int id) noexcept(false) {
  this->checkConnection();

  try {
    Statement *stmt = m_conn->createStatement();
    stmt->setSQL(
        "SELECT cycle_id, ccs_configuration_id FROM ecal_ccs_cycle "
        "WHERE cycle_id = :1 ");
    stmt->setInt(1, id);
    ResultSet *rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
      m_ccs_config_id = rset->getInt(2);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("ODCCSCycle::fetchID:  " + e.getMessage()));
  }
}

void ODCCSCycle::fetchData(ODCCSCycle *result) noexcept(false) {
  this->checkConnection();
  result->clear();

  if (result->getId() == 0) {
    throw(std::runtime_error("ODCCSConfig::fetchData(): no Id defined for this ODCCSConfig "));
  }

  try {
    m_readStmt->setSQL(
        "SELECT  ccs_configuration_id FROM ecal_ccs_cycle "
        "WHERE cycle_id = :1 ");

    m_readStmt->setInt(1, result->getId());
    ResultSet *rset = m_readStmt->executeQuery();

    rset->next();

    result->setCCSConfigurationID(rset->getInt(1));

  } catch (SQLException &e) {
    throw(std::runtime_error("ODCCSCycle::fetchData():  " + e.getMessage()));
  }
}

void ODCCSCycle::insertConfig() noexcept(false) {
  try {
    prepareWrite();
    writeDB();
    terminateWriteStatement();
  } catch (SQLException &e) {
    throw(std::runtime_error("EcalCondDBInterface::insertDataSet:  Unknown exception caught"));
  }
}
