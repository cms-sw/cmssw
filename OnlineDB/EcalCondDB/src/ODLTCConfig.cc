#include <fstream>
#include <iostream>
#include <cstdio>
#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODLTCConfig.h"

using namespace std;
using namespace oracle::occi;

ODLTCConfig::ODLTCConfig() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;
  m_config_tag = "";
  m_size = 0;

  m_ID = 0;
  clear();
}

ODLTCConfig::~ODLTCConfig() {
  //  delete [] m_ltc_clob;
}

int ODLTCConfig::fetchNextId() noexcept(false) {
  int result = 0;
  try {
    this->checkConnection();

    m_readStmt = m_conn->createStatement();
    m_readStmt->setSQL("select ecal_ltc_config_sq.NextVal from dual");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next()) {
      result = rset->getInt(1);
    }
    m_conn->terminateStatement(m_readStmt);
    return result;

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODLTCConfig::fetchNextId():  ") + e.getMessage()));
  }
}

void ODLTCConfig::prepareWrite() noexcept(false) {
  this->checkConnection();

  int next_id = fetchNextId();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO ECAL_LTC_CONFIGURATION (ltc_configuration_id, ltc_tag, "
        " LTC_CONFIGURATION_file, "
        " Configuration ) "
        "VALUES (:1, :2, :3, :4 )");
    m_writeStmt->setInt(1, next_id);
    m_writeStmt->setString(2, this->getConfigTag());
    m_writeStmt->setString(3, getLTCConfigurationFile());

    // and now the clob
    oracle::occi::Clob clob(m_conn);
    clob.setEmpty();
    m_writeStmt->setClob(4, clob);
    m_writeStmt->executeUpdate();
    m_ID = next_id;

    m_conn->terminateStatement(m_writeStmt);
    std::cout << "LTC Clob inserted into CONFIGURATION with id=" << next_id << std::endl;

    // now we read and update it
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "SELECT Configuration FROM ECAL_LTC_CONFIGURATION WHERE"
        " ltc_configuration_id=:1 FOR UPDATE");

    std::cout << "updating the clob 0" << std::endl;

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODLTCConfig::prepareWrite():  ") + e.getMessage()));
  }

  std::cout << "updating the clob 1 " << std::endl;
}

void ODLTCConfig::setParameters(const std::map<string, string>& my_keys_map) {
  // parses the result of the XML parser that is a map of
  // string string with variable name variable value

  for (std::map<std::string, std::string>::const_iterator ci = my_keys_map.begin(); ci != my_keys_map.end(); ci++) {
    if (ci->first == "LTC_CONFIGURATION_ID")
      setConfigTag(ci->second);
    if (ci->first == "Configuration") {
      std::string fname = ci->second;
      string str3;
      size_t pos, pose;

      pos = fname.find('=');  // position of "live" in str
      pose = fname.size();    // position of "]" in str
      str3 = fname.substr(pos + 1, pose - pos - 2);

      cout << "fname=" << fname << " and reduced is: " << str3 << endl;
      setLTCConfigurationFile(str3);

      // here we must open the file and read the LTC Clob
      std::cout << "Going to read LTC file: " << fname << endl;

      ifstream inpFile;
      inpFile.open(str3.c_str());

      // tell me size of file
      int bufsize = 0;
      inpFile.seekg(0, ios::end);
      bufsize = inpFile.tellg();
      std::cout << " bufsize =" << bufsize << std::endl;
      // set file pointer to start again
      inpFile.seekg(0, ios::beg);

      m_size = bufsize;

      inpFile.close();
    }
  }
}

void ODLTCConfig::writeDB() noexcept(false) {
  std::cout << "updating the clob " << std::endl;

  try {
    m_writeStmt->setInt(1, m_ID);
    ResultSet* rset = m_writeStmt->executeQuery();

    rset->next();
    oracle::occi::Clob clob = rset->getClob(1);

    cout << "Opening the clob in read write mode" << endl;

    std::cout << "Populating the clob" << endl;

    populateClob(clob, getLTCConfigurationFile(), m_size);
    int clobLength = clob.length();
    cout << "Length of the clob is: " << clobLength << endl;
    // clob.close ();

    m_writeStmt->executeUpdate();

    m_writeStmt->closeResultSet(rset);

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODLTCConfig::writeDB():  ") + e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODLTCConfig::writeDB:  Failed to write"));
  }
}

void ODLTCConfig::clear() {
  //  strcpy((char *)m_ltc_clob, "");

  m_ltc_file = "";
}

void ODLTCConfig::fetchData(ODLTCConfig* result) noexcept(false) {
  this->checkConnection();
  result->clear();
  if (result->getId() == 0 && result->getConfigTag().empty()) {
    throw(std::runtime_error("ODLTCConfig::fetchData(): no Id defined for this ODLTCConfig "));
  }

  try {
    m_readStmt->setSQL(
        "SELECT *   "
        "FROM ECAL_LTC_CONFIGURATION  "
        " where (ltc_configuration_id = :1  or LTC_tag=:2 )");
    m_readStmt->setInt(1, result->getId());
    m_readStmt->setString(2, result->getConfigTag());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();
    // 1 is the id and 2 is the config tag

    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));
    result->setLTCConfigurationFile(rset->getString(3));

    Clob clob = rset->getClob(4);
    cout << "Opening the clob in Read only mode" << endl;
    clob.open(OCCI_LOB_READONLY);
    int clobLength = clob.length();
    cout << "Length of the clob is: " << clobLength << endl;
    m_size = clobLength;
    unsigned char* buffer = readClob(clob, clobLength);
    clob.close();
    cout << "the clob buffer is:" << endl;
    for (int i = 0; i < clobLength; ++i)
      cout << (char)buffer[i];
    cout << endl;

    result->setLTCClob(buffer);

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODLTCConfig::fetchData():  ") + e.getMessage()));
  }
}

int ODLTCConfig::fetchID() noexcept(false) {
  if (m_ID != 0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL(
        "SELECT ltc_configuration_id FROM ecal_ltc_configuration "
        "WHERE  ltc_tag=:ltc_tag ");

    stmt->setString(1, getConfigTag());

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODLTCConfig::fetchID:  ") + e.getMessage()));
  }

  return m_ID;
}
