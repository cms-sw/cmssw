#include <stdexcept>
#include <cstdlib>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODScanConfig.h"

using namespace std;
using namespace oracle::occi;

ODScanConfig::ODScanConfig() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;
  m_config_tag = "";
  m_ID = 0;
  clear();
}

ODScanConfig::~ODScanConfig() {}

void ODScanConfig::clear() {
  m_type_id = 0;
  m_type = "";
  m_from_val = 0;
  m_to_val = 0;
  m_step = 0;
}

int ODScanConfig::fetchNextId() noexcept(false) {
  int result = 0;
  try {
    this->checkConnection();

    m_readStmt = m_conn->createStatement();
    m_readStmt->setSQL("select ecal_scan_config_sq.NextVal from dual");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next()) {
      result = rset->getInt(1);
    }
    m_conn->terminateStatement(m_readStmt);
    return result;

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODScanConfig::fetchNextId():  ") + e.getMessage()));
  }
}

void ODScanConfig::setParameters(const std::map<string, string>& my_keys_map) {
  // parses the result of the XML parser that is a map of
  // string string with variable name variable value

  for (std::map<std::string, std::string>::const_iterator ci = my_keys_map.begin(); ci != my_keys_map.end(); ci++) {
    if (ci->first == "SCAN_ID")
      setConfigTag(ci->second);
    if (ci->first == "TYPE_ID")
      setTypeId(atoi(ci->second.c_str()));
    if (ci->first == "TYPE" || ci->first == "SCAN_TYPE")
      setScanType(ci->second);
    if (ci->first == "FROM" || ci->first == "FROM_VAL")
      setFromVal(atoi(ci->second.c_str()));
    if (ci->first == "TO" || ci->first == "TO_VAL")
      setToVal(atoi(ci->second.c_str()));
    if (ci->first == "STEP")
      setStep(atoi(ci->second.c_str()));
  }
}

void ODScanConfig::prepareWrite() noexcept(false) {
  this->checkConnection();
  int next_id = fetchNextId();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO ECAL_scan_dat ( scan_id, scan_tag ,"
        "   type_id, scan_type , FROM_VAL , TO_VAL, STEP )"
        " VALUES ( :1, :2, :3, :4, :5, :6, :7)");
    m_writeStmt->setInt(1, next_id);
    m_ID = next_id;

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODScanConfig::prepareWrite():  ") + e.getMessage()));
  }
}

void ODScanConfig::writeDB() noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  try {
    // number 1 is the id
    m_writeStmt->setString(2, this->getConfigTag());

    m_writeStmt->setInt(3, this->getTypeId());
    m_writeStmt->setString(4, this->getScanType());
    m_writeStmt->setInt(5, this->getFromVal());
    m_writeStmt->setInt(6, this->getToVal());
    m_writeStmt->setInt(7, this->getStep());

    m_writeStmt->executeUpdate();

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODScanConfig::writeDB():  ") + e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODScanConfig::writeDB:  Failed to write"));
  }
}

void ODScanConfig::fetchData(ODScanConfig* result) noexcept(false) {
  this->checkConnection();
  result->clear();
  if (result->getId() == 0 && (result->getConfigTag().empty())) {
    throw(std::runtime_error("ODScanConfig::fetchData(): no Id defined for this ODScanConfig "));
  }

  try {
    m_readStmt->setSQL(
        "SELECT * "
        "FROM ECAL_SCAN_DAT "
        " where (scan_id = :1  or scan_tag=:2 )");
    m_readStmt->setInt(1, result->getId());
    m_readStmt->setString(2, result->getConfigTag());

    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    // id 1 is the scan_id
    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));
    result->setTypeId(rset->getInt(3));
    result->setScanType(rset->getString(4));
    result->setFromVal(rset->getInt(5));
    result->setToVal(rset->getInt(6));
    result->setStep(rset->getInt(7));

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODScanConfig::fetchData():  ") + e.getMessage()));
  }
}

int ODScanConfig::fetchID() noexcept(false) {
  // Return from memory if available
  if (m_ID != 0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL(
        "SELECT scan_id FROM ecal_scan_dat "
        "WHERE scan_tag=:scan_tag ");

    stmt->setString(1, getConfigTag());

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODScanConfig::fetchID:  ") + e.getMessage()));
  }

  return m_ID;
}
