#include <stdexcept>
#include <string>
#include <cstring>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include <cstdlib>
#include "OnlineDB/EcalCondDB/interface/FEConfigOddWeightInfo.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;

FEConfigOddWeightInfo::FEConfigOddWeightInfo() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;
  m_config_tag = "";
  m_version = 0;
  m_ID = 0;
  clear();
}

void FEConfigOddWeightInfo::clear() { m_ngr = 0; }

FEConfigOddWeightInfo::~FEConfigOddWeightInfo() {}

int FEConfigOddWeightInfo::fetchNextId() noexcept(false) {
  int result = 0;
  try {
    this->checkConnection();

    m_readStmt = m_conn->createStatement();
    m_readStmt->setSQL("select FE_CONFIG_WEIGHT2GROUP_SQ.NextVal from DUAL ");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next()) {
      result = rset->getInt(1);
    }
    result++;
    m_conn->terminateStatement(m_readStmt);
    return result;

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("FEConfigOddWeightInfo::fetchNextId():  ") + e.getMessage()));
  }
}

void FEConfigOddWeightInfo::prepareWrite() noexcept(false) {
  this->checkConnection();

  int next_id = 0;
  if (getId() == 0) {
    next_id = fetchNextId();
  }

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO " + getTable() +
                        " ( wei2_conf_id, tag, number_of_groups) "
                        " VALUES ( :1, :2, :3 ) ");

    m_writeStmt->setInt(1, next_id);
    m_ID = next_id;

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("FEConfigOddWeightInfo::prepareWrite():  ") + e.getMessage()));
  }
}

void FEConfigOddWeightInfo::setParameters(const std::map<string, string>& my_keys_map) {
  // parses the result of the XML parser that is a map of
  // string string with variable name variable value

  for (std::map<std::string, std::string>::const_iterator ci = my_keys_map.begin(); ci != my_keys_map.end(); ci++) {
    if (ci->first == "TAG")
      setConfigTag(ci->second);
    if (ci->first == "NUMBER_OF_GROUPS")
      setNumberOfGroups(atoi(ci->second.c_str()));
  }
}

void FEConfigOddWeightInfo::writeDB() noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  try {
    // number 1 is the id
    m_writeStmt->setString(2, this->getConfigTag());
    m_writeStmt->setInt(3, this->getNumberOfGroups());

    m_writeStmt->executeUpdate();

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("FEConfigOddWeightInfo::writeDB():  ") + e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("FEConfigOddWeightInfo::writeDB:  Failed to write"));
  }
}

void FEConfigOddWeightInfo::fetchData(FEConfigOddWeightInfo* result) noexcept(false) {
  this->checkConnection();
  result->clear();
  if (result->getId() == 0 && (result->getConfigTag().empty())) {
    throw(std::runtime_error("FEConfigOddWeightInfo::fetchData(): no Id defined for this FEConfigOddWeightInfo "));
  }

  try {
    DateHandler dh(m_env, m_conn);

    m_readStmt->setSQL("SELECT wei2_conf_id, tag, number_of_groups, db_timestamp  FROM " + getTable() +
                       " where ( wei2_conf_id= :1 or (tag=:2 ) )");
    m_readStmt->setInt(1, result->getId());
    m_readStmt->setString(2, result->getConfigTag());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    // 1 is the id and 2 is the config tag and 3 is the version

    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));
    result->setNumberOfGroups(rset->getInt(3));
    Date dbdate = rset->getDate(4);
    result->setDBTime(dh.dateToTm(dbdate));

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("FEConfigOddWeightInfo::fetchData():  ") + e.getMessage()));
  }
}

void FEConfigOddWeightInfo::fetchLastData(FEConfigOddWeightInfo* result) noexcept(false) {
  this->checkConnection();
  result->clear();
  try {
    DateHandler dh(m_env, m_conn);

    m_readStmt->setSQL("SELECT wei2_conf_id, tag, number_of_groups, db_timestamp FROM " + getTable() +
                       " where   wei2_conf_id = ( select max( wei2_conf_id) from " + getTable() + " ) ");
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));
    result->setNumberOfGroups(rset->getInt(3));
    Date dbdate = rset->getDate(4);
    result->setDBTime(dh.dateToTm(dbdate));

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("FEConfigOddWeightInfo::fetchData():  ") + e.getMessage()));
  }
}

int FEConfigOddWeightInfo::fetchID() noexcept(false) {
  // Return from memory if available
  if (m_ID != 0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT wei2_conf_id FROM " + getTable() + " WHERE  tag=:1 ");

    stmt->setString(1, getConfigTag());

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("FEConfigOddWeightInfo::fetchID:  ") + e.getMessage()));
  }

  return m_ID;
}

void FEConfigOddWeightInfo::setByID(int id) noexcept(false) {
  this->checkConnection();

  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("SELECT wei2_conf_id, tag, number_of_groups, db_timestamp  FROM " + getTable() +
                 " WHERE wei2_conf_id = :1");
    stmt->setInt(1, id);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      this->setId(rset->getInt(1));
      this->setConfigTag(rset->getString(2));
      this->setNumberOfGroups(rset->getInt(3));
      Date dbdate = rset->getDate(4);
      this->setDBTime(dh.dateToTm(dbdate));
    } else {
      throw(std::runtime_error("FEConfigOddWeightInfo::setByID:  Given config_id is not in the database"));
    }

    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("FEConfigOddWeightInfo::setByID:  ") + e.getMessage()));
  }
}
