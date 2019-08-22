#include <stdexcept>
#include <string>
#include <cstring>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include <cstdlib>
#include "OnlineDB/EcalCondDB/interface/FEConfigLUTInfo.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;

FEConfigLUTInfo::FEConfigLUTInfo() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;
  m_config_tag = "";
  m_version = 0;
  m_ID = 0;
  clear();
}

void FEConfigLUTInfo::clear() { m_iov_id = 0; }

FEConfigLUTInfo::~FEConfigLUTInfo() {}

int FEConfigLUTInfo::fetchNextId() noexcept(false) {
  int result = 0;
  try {
    this->checkConnection();

    m_readStmt = m_conn->createStatement();
    m_readStmt->setSQL("select FE_CONFIG_LUT_SQ.NextVal from DUAL ");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next()) {
      result = rset->getInt(1);
    }
    result++;
    m_conn->terminateStatement(m_readStmt);
    return result;

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("FEConfigLUTInfo::fetchNextId():  ") + e.getMessage()));
  }
}

void FEConfigLUTInfo::prepareWrite() noexcept(false) {
  this->checkConnection();

  int next_id = 0;
  if (getId() == 0) {
    next_id = fetchNextId();
  }

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO " + getTable() +
                        " ( lut_conf_id, tag, version, number_of_groups) "
                        " VALUES ( :1, :2, :3 , :4) ");

    m_writeStmt->setInt(1, next_id);
    m_ID = next_id;

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("FEConfigLUTInfo::prepareWrite():  ") + e.getMessage()));
  }
}

void FEConfigLUTInfo::setParameters(const std::map<string, string>& my_keys_map) {
  // parses the result of the XML parser that is a map of
  // string string with variable name variable value

  for (std::map<std::string, std::string>::const_iterator ci = my_keys_map.begin(); ci != my_keys_map.end(); ci++) {
    if (ci->first == "VERSION")
      setVersion(atoi(ci->second.c_str()));
    if (ci->first == "TAG")
      setConfigTag(ci->second);
    if (ci->first == "NUMBER_OF_GROUPS")
      setNumberOfGroups(atoi(ci->second.c_str()));
  }
}

void FEConfigLUTInfo::writeDB() noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  try {
    // number 1 is the id
    m_writeStmt->setString(2, this->getConfigTag());
    m_writeStmt->setInt(3, this->getVersion());
    m_writeStmt->setInt(4, this->getNumberOfGroups());

    m_writeStmt->executeUpdate();

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("FEConfigLUTInfo::writeDB():  ") + e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("FEConfigLUTInfo::writeDB:  Failed to write"));
  }
}

void FEConfigLUTInfo::fetchData(FEConfigLUTInfo* result) noexcept(false) {
  this->checkConnection();
  result->clear();
  if (result->getId() == 0 && (result->getConfigTag().empty())) {
    throw(std::runtime_error("FEConfigLUTInfo::fetchData(): no Id defined for this FEConfigLUTInfo "));
  }

  try {
    DateHandler dh(m_env, m_conn);

    m_readStmt->setSQL("SELECT * FROM " + getTable() + " where ( lut_conf_id= :1 or (tag=:2 AND version=:3 ) )");
    m_readStmt->setInt(1, result->getId());
    m_readStmt->setString(2, result->getConfigTag());
    m_readStmt->setInt(3, result->getVersion());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    // 1 is the id and 2 is the config tag and 3 is the version

    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));
    result->setVersion(rset->getInt(3));
    result->setNumberOfGroups(rset->getInt(4));
    Date dbdate = rset->getDate(5);
    result->setDBTime(dh.dateToTm(dbdate));

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("FEConfigLUTInfo::fetchData():  ") + e.getMessage()));
  }
}

void FEConfigLUTInfo::fetchLastData(FEConfigLUTInfo* result) noexcept(false) {
  this->checkConnection();
  result->clear();
  try {
    DateHandler dh(m_env, m_conn);

    m_readStmt->setSQL("SELECT * FROM " + getTable() + " where   lut_conf_id = ( select max( lut_conf_id) from " +
                       getTable() + " ) ");
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));
    result->setVersion(rset->getInt(3));
    result->setNumberOfGroups(rset->getInt(4));
    Date dbdate = rset->getDate(5);
    result->setDBTime(dh.dateToTm(dbdate));

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("FEConfigLUTInfo::fetchData():  ") + e.getMessage()));
  }
}

int FEConfigLUTInfo::fetchID() noexcept(false) {
  // Return from memory if available
  if (m_ID != 0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT lut_conf_id FROM " + getTable() + " WHERE  tag=:1 and version=:2 ");

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
    throw(std::runtime_error(std::string("FEConfigLUTInfo::fetchID:  ") + e.getMessage()));
  }

  return m_ID;
}

void FEConfigLUTInfo::setByID(int id) noexcept(false) {
  this->checkConnection();

  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("SELECT * FROM fe_config_lut_info WHERE lut_conf_id = :1");
    stmt->setInt(1, id);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      this->setId(rset->getInt(1));
      this->setConfigTag(rset->getString(2));
      this->setVersion(rset->getInt(3));
      this->setNumberOfGroups(rset->getInt(4));
      Date dbdate = rset->getDate(5);
      this->setDBTime(dh.dateToTm(dbdate));
    } else {
      throw(std::runtime_error("FEConfigLUTInfo::setByID:  Given config_id is not in the database"));
    }

    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("FEConfigLUTInfo::setByID:  ") + e.getMessage()));
  }
}
