#include <fstream>
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODTTCciConfig.h"

using namespace std;
using namespace oracle::occi;

ODTTCciConfig::ODTTCciConfig() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;
  m_config_tag = "";
  m_configuration_script = "";
  m_configuration_script_params = "";
  m_ID = 0;
  clear();
  m_size = 0;
}

void ODTTCciConfig::clear() {
  std::cout << "entering clear" << std::endl;
  m_ttcci_file = "";
  m_configuration_script = "";
  m_configuration_script_params = "";
  m_trg_mode = "";
  m_trg_sleep = 0;
}

ODTTCciConfig::~ODTTCciConfig() {}

int ODTTCciConfig::fetchNextId() noexcept(false) {
  int result = 0;
  try {
    this->checkConnection();

    m_readStmt = m_conn->createStatement();
    m_readStmt->setSQL("select ecal_ttcci_config_sq.NextVal from dual");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next()) {
      result = rset->getInt(1);
    }
    m_conn->terminateStatement(m_readStmt);
    return result;

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODTTCciConfig::fetchNextId():  ") + e.getMessage()));
  }
}

void ODTTCciConfig::prepareWrite() noexcept(false) {
  this->checkConnection();

  int next_id = fetchNextId();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO ECAL_TTCci_CONFIGURATION (ttcci_configuration_id, ttcci_tag, "
        " TTCCI_configuration_file, TRG_MODE, TRG_SLEEP, Configuration, configuration_script, "
        "configuration_script_params  ) "
        "VALUES (:1, :2, :3, :4, :5, :6, :7, :8  )");
    m_writeStmt->setInt(1, next_id);
    m_writeStmt->setString(2, getConfigTag());
    m_writeStmt->setString(3, getTTCciConfigurationFile());
    m_writeStmt->setString(4, getTrgMode());
    m_writeStmt->setInt(5, getTrgSleep());
    m_writeStmt->setString(7, getConfigurationScript());
    m_writeStmt->setString(8, getConfigurationScriptParams());

    // and now the clob
    oracle::occi::Clob clob(m_conn);
    clob.setEmpty();
    m_writeStmt->setClob(6, clob);
    m_writeStmt->executeUpdate();
    m_ID = next_id;

    m_conn->terminateStatement(m_writeStmt);
    std::cout << "TTCci Clob inserted into CONFIGURATION with id=" << next_id << std::endl;

    // now we read and update it
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "SELECT Configuration FROM ECAL_TTCci_CONFIGURATION WHERE"
        " ttcci_configuration_id=:1 FOR UPDATE");

    std::cout << "updating the clob 0" << std::endl;

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODTTCciConfig::prepareWrite():  ") + e.getMessage()));
  }

  std::cout << "updating the clob 1 " << std::endl;
}

void ODTTCciConfig::setParameters(const std::map<string, string>& my_keys_map) {
  // parses the result of the XML parser that is a map of
  // string string with variable name variable value

  for (std::map<std::string, std::string>::const_iterator ci = my_keys_map.begin(); ci != my_keys_map.end(); ci++) {
    if (ci->first == "TRG_MODE")
      setTrgMode(ci->second);
    if (ci->first == "TRG_SLEEP")
      setTrgSleep(atoi(ci->second.c_str()));
    if (ci->first == "TTCci_CONFIGURATION_ID")
      setConfigTag(ci->second);
    if (ci->first == "CONFIGURATION_SCRIPT")
      setConfigurationScript(ci->second);
    if (ci->first == "CONFIGURATION_SCRIPT_PARAMS")
      setConfigurationScriptParams(ci->second);
    if (ci->first == "CONFIGURATION_SCRIPT_PARAMETERS")
      setConfigurationScriptParams(ci->second);
    if (ci->first == "Configuration") {
      std::string fname = ci->second;
      string str3;
      size_t pos, pose;

      pos = fname.find('=');  // position of "live" in str
      pose = fname.size();    // position of "]" in str
      str3 = fname.substr(pos + 1, pose - pos - 2);

      cout << "fname=" << fname << " and reduced is: " << str3 << endl;
      setTTCciConfigurationFile(str3);

      // here we must open the file and read the LTC Clob
      std::cout << "Going to read file: " << str3 << endl;

      ifstream inpFile;
      inpFile.open(str3.c_str());

      // tell me size of file
      int bufsize = 0;
      inpFile.seekg(0, ios::end);
      bufsize = inpFile.tellg();
      std::cout << " bufsize =" << bufsize << std::endl;
      // set file pointer to start again
      inpFile.seekg(0, ios::beg);

      inpFile.close();
      m_size = bufsize;
    }
  }
}

void ODTTCciConfig::writeDB() noexcept(false) {
  std::cout << "updating the clob 2" << std::endl;

  try {
    m_writeStmt->setInt(1, m_ID);
    ResultSet* rset = m_writeStmt->executeQuery();

    while (rset->next()) {
      oracle::occi::Clob clob = rset->getClob(1);
      cout << "Opening the clob in read write mode" << endl;
      cout << "Populating the clob" << endl;
      populateClob(clob, getTTCciConfigurationFile(), m_size);
      int clobLength = clob.length();
      cout << "Length of the clob is: " << clobLength << endl;
      //        clob.close ();
    }

    m_writeStmt->executeUpdate();

    m_writeStmt->closeResultSet(rset);

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODTTCciConfig::writeDB():  ") + e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODTTCciConfig::writeDB:  Failed to write"));
  }
}

void ODTTCciConfig::fetchData(ODTTCciConfig* result) noexcept(false) {
  this->checkConnection();
  result->clear();
  if (result->getId() == 0 && (result->getConfigTag().empty())) {
    throw(std::runtime_error("ODTTCciConfig::fetchData(): no Id defined for this ODTTCciConfig "));
  }

  try {
    m_readStmt->setSQL(
        "SELECT * "
        "FROM ECAL_TTCci_CONFIGURATION  "
        " where ( ttcci_configuration_id = :1 or ttcci_tag=:2 )");
    m_readStmt->setInt(1, result->getId());
    m_readStmt->setString(2, result->getConfigTag());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();
    // 1 is the id and 2 is the config tag

    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));

    result->setTTCciConfigurationFile(rset->getString(3));
    result->setTrgMode(rset->getString(4));
    result->setTrgSleep(rset->getInt(5));

    result->setConfigurationScript(rset->getString(7));
    result->setConfigurationScriptParams(rset->getString(8));

    Clob clob = rset->getClob(6);
    cout << "Opening the clob in Read only mode" << endl;
    clob.open(OCCI_LOB_READONLY);
    int clobLength = clob.length();
    cout << "Length of the clob is: " << clobLength << endl;
    m_size = clobLength;
    unsigned char* buffer = readClob(clob, m_size);
    clob.close();
    cout << "the clob buffer is:" << endl;
    for (int i = 0; i < clobLength; ++i)
      cout << (char)buffer[i];
    cout << endl;

    result->setTTCciClob(buffer);

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODTTCciConfig::fetchData():  ") + e.getMessage()));
  }
}

int ODTTCciConfig::fetchID() noexcept(false) {
  if (m_ID != 0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL(
        "SELECT ttcci_configuration_id FROM ecal_ttcci_configuration "
        "WHERE  ttcci_tag=:ttcci_tag ");

    stmt->setString(1, getConfigTag());

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODTTCciConfig::fetchID:  ") + e.getMessage()));
  }
  return m_ID;
}
