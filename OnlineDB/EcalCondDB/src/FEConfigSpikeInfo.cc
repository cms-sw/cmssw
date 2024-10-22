#include <stdexcept>
#include <string>
#include <cstring>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include <cstdlib>
#include "OnlineDB/EcalCondDB/interface/FEConfigSpikeInfo.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;

FEConfigSpikeInfo::FEConfigSpikeInfo() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;
  m_config_tag = "";
  m_version = 0;
  m_ID = 0;
  clear();
}

void FEConfigSpikeInfo::clear() {}

FEConfigSpikeInfo::~FEConfigSpikeInfo() {}

int FEConfigSpikeInfo::fetchNextId() noexcept(false) {
  int result = 0;
  try {
    this->checkConnection();

    m_readStmt = m_conn->createStatement();
    m_readStmt->setSQL("select FE_CONFIG_SPI_SQ.NextVal from DUAL ");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next()) {
      result = rset->getInt(1);
    }
    result++;
    m_conn->terminateStatement(m_readStmt);
    return result;

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("FEConfigSpikeInfo::fetchNextId():  ") + e.getMessage()));
  }
}

void FEConfigSpikeInfo::prepareWrite() noexcept(false) {
  this->checkConnection();

  int next_id = 0;
  if (getId() == 0) {
    next_id = fetchNextId();
  }

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO " + getTable() +
                        " ( spi_conf_id, tag, version ) "
                        " VALUES ( :1, :2, :3 ) ");

    m_writeStmt->setInt(1, next_id);
    m_ID = next_id;

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("FEConfigSpikeInfo::prepareWrite():  ") + e.getMessage()));
  }
}

void FEConfigSpikeInfo::setParameters(const std::map<string, string>& my_keys_map) {
  // parses the result of the XML parser that is a map of
  // string string with variable name variable value

  for (std::map<std::string, std::string>::const_iterator ci = my_keys_map.begin(); ci != my_keys_map.end(); ci++) {
    if (ci->first == "VERSION")
      setVersion(atoi(ci->second.c_str()));
    if (ci->first == "TAG")
      setConfigTag(ci->second);
  }
}

void FEConfigSpikeInfo::writeDB() noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  try {
    // number 1 is the id
    m_writeStmt->setString(2, this->getConfigTag());
    m_writeStmt->setInt(3, this->getVersion());

    m_writeStmt->executeUpdate();

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("FEConfigSpikeInfo::writeDB():  ") + e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("FEConfigSpikeInfo::writeDB:  Failed to write"));
  }
}

void FEConfigSpikeInfo::fetchData(FEConfigSpikeInfo* result) noexcept(false) {
  this->checkConnection();
  result->clear();
  if (result->getId() == 0 && (result->getConfigTag().empty())) {
    throw(std::runtime_error("FEConfigSpikeInfo::fetchData(): no Id defined for this FEConfigSpikeInfo "));
  }

  try {
    DateHandler dh(m_env, m_conn);

    m_readStmt->setSQL("SELECT * FROM " + getTable() + " where ( spi_conf_id= :1 or (tag=:2 AND version=:3 ) )");
    m_readStmt->setInt(1, result->getId());
    m_readStmt->setString(2, result->getConfigTag());
    m_readStmt->setInt(3, result->getVersion());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    // 1 is the id and 2 is the config tag and 3 is the version

    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));
    result->setVersion(rset->getInt(3));
    Date dbdate = rset->getDate(4);
    result->setDBTime(dh.dateToTm(dbdate));

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("FEConfigSpikeInfo::fetchData():  ") + e.getMessage()));
  }
}

void FEConfigSpikeInfo::fetchLastData(FEConfigSpikeInfo* result) noexcept(false) {
  this->checkConnection();
  result->clear();
  try {
    DateHandler dh(m_env, m_conn);

    m_readStmt->setSQL("SELECT * FROM " + getTable() + " where   spi_conf_id = ( select max( spi_conf_id) from " +
                       getTable() + " ) ");
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));
    result->setVersion(rset->getInt(3));
    Date dbdate = rset->getDate(4);
    result->setDBTime(dh.dateToTm(dbdate));

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("FEConfigSpikeInfo::fetchData():  ") + e.getMessage()));
  }
}

int FEConfigSpikeInfo::fetchID() noexcept(false) {
  // Return from memory if available
  if (m_ID != 0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT spi_conf_id FROM " + getTable() + " WHERE  tag=:1 and version=:2 ");

    stmt->setString(1, getConfigTag());
    stmt->setInt(2, getVersion());

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("FEConfigSpikeInfo::fetchID:  ") + e.getMessage()));
  }

  return m_ID;
}

void FEConfigSpikeInfo::setByID(int id) noexcept(false) {
  this->checkConnection();

  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("SELECT * FROM " + getTable() + " WHERE spi_conf_id = :1");
    stmt->setInt(1, id);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      this->setId(rset->getInt(1));
      this->setConfigTag(rset->getString(2));
      this->setVersion(rset->getInt(3));
      Date dbdate = rset->getDate(4);
      this->setDBTime(dh.dateToTm(dbdate));
    } else {
      throw(std::runtime_error("FEConfigSpikeInfo::setByID: Given spi_conf_id is not in the database"));
    }

    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("FEConfigSpikeInfo::setByID:  ") + e.getMessage()));
  }
}
