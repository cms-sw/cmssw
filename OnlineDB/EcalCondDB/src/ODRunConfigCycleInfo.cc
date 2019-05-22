#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODRunConfigCycleInfo.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;

ODRunConfigCycleInfo::ODRunConfigCycleInfo() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_ID = 0;

  m_sequence_id = 0;
  m_cycle_num = 0;
  m_tag = "";
  m_description = "";
}

ODRunConfigCycleInfo::~ODRunConfigCycleInfo() {}

void ODRunConfigCycleInfo::clear() {
  m_sequence_id = 0;
  m_cycle_num = 0;
  m_tag = "";
  m_description = "";
}

void ODRunConfigCycleInfo::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO ECAL_CYCLE_DAT ( sequence_id , cycle_num, tag, description ) "
        "VALUES (:1, :2, :3 , :4 )");

  } catch (SQLException& e) {
    throw(std::runtime_error("ODRunConfigCycleInfo::prepareWrite():  " + e.getMessage()));
  }
}

void ODRunConfigCycleInfo::writeDB() noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  // Validate the data, use infinity-till convention
  DateHandler dh(m_env, m_conn);

  try {
    m_writeStmt->setInt(1, this->getSequenceID());
    m_writeStmt->setInt(2, this->getCycleNumber());
    m_writeStmt->setString(3, this->getTag());
    m_writeStmt->setString(4, this->getDescription());
    m_writeStmt->executeUpdate();

  } catch (SQLException& e) {
    throw(std::runtime_error("ODRunConfigCycleInfo::writeDB:  " + e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODRunConfigCycleInfo::writeDB:  Failed to write"));
  }

  cout << "ODRunConfigCycleInfo::writeDB>> done inserting ODRunConfigCycleInfo with id=" << m_ID << endl;
}

int ODRunConfigCycleInfo::fetchID() noexcept(false) {
  // Return from memory if available
  if (m_ID > 0) {
    return m_ID;
  }

  this->checkConnection();

  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL(
        "SELECT cycle_id from ECAL_cycle_DAT "
        "WHERE sequence_id = :id1 "
        " and cycle_num = :id2  ");
    stmt->setInt(1, m_sequence_id);
    stmt->setInt(2, m_cycle_num);

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("ODRunConfigCycleInfo::fetchID:  " + e.getMessage()));
  }
  setByID(m_ID);

  return m_ID;
}

int ODRunConfigCycleInfo::fetchIDLast() noexcept(false) {
  this->checkConnection();

  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT max(cycle_id) FROM ecal_cycle_dat ");
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("ODRunConfigCycleInfo::fetchIDLast:  " + e.getMessage()));
  }

  setByID(m_ID);
  return m_ID;
}

void ODRunConfigCycleInfo::setByID(int id) noexcept(false) {
  this->checkConnection();

  DateHandler dh(m_env, m_conn);

  cout << "ODRunConfigCycleInfo::setByID called for id " << id << endl;

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("SELECT sequence_id , cycle_num , tag , description FROM ECAL_cycle_DAT WHERE cycle_id = :1 ");
    stmt->setInt(1, id);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      m_sequence_id = rset->getInt(1);
      m_cycle_num = rset->getInt(2);
      m_tag = rset->getString(3);
      m_description = rset->getString(4);
      m_ID = id;
    } else {
      throw(std::runtime_error("ODRunConfigCycleInfo::setByID:  Given cycle_id is not in the database"));
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("ODRunConfigCycleInfo::setByID:  " + e.getMessage()));
  }
}

void ODRunConfigCycleInfo::fetchData(ODRunConfigCycleInfo* result) noexcept(false) {
  this->checkConnection();
  result->clear();
  if (result->getId() == 0) {
    throw(std::runtime_error("ODRunConfigCycleInfo::fetchData(): no Id defined for this ODRunConfigCycleInfo "));
  }

  try {
    m_readStmt->setSQL("SELECT sequence_id , cycle_num , tag , description FROM ECAL_cycle_DAT WHERE cycle_id = :1 ");

    m_readStmt->setInt(1, result->getId());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setSequenceID(rset->getInt(1));
    result->setCycleNumber(rset->getInt(2));
    result->setTag(rset->getString(3));
    result->setDescription(rset->getString(4));

  } catch (SQLException& e) {
    throw(std::runtime_error("ODRunConfigCycleInfo::fetchData():  " + e.getMessage()));
  }
}

void ODRunConfigCycleInfo::insertConfig() noexcept(false) {
  try {
    prepareWrite();
    writeDB();
    m_conn->commit();
    terminateWriteStatement();
  } catch (std::runtime_error& e) {
    m_conn->rollback();
    throw(e);
  } catch (...) {
    m_conn->rollback();
    throw(std::runtime_error("EcalCondDBInterface::insertDataSet:  Unknown exception caught"));
  }
}
