#include <stdexcept>
#include <string>
#include <cstdlib>

#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODTTCFConfig.h"

using namespace std;
using namespace oracle::occi;

ODTTCFConfig::ODTTCFConfig() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;
  m_config_tag = "";
  m_ID = 0;
  clear();
  m_size = 0;
}

void ODTTCFConfig::clear() {}

ODTTCFConfig::~ODTTCFConfig() {}

int ODTTCFConfig::fetchNextId() noexcept(false) {
  int result = 0;
  try {
    this->checkConnection();
    std::cout << "going to fetch new id for TTCF 1" << endl;
    m_readStmt = m_conn->createStatement();
    m_readStmt->setSQL("select ecal_ttcf_config_sq.NextVal from dual");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next()) {
      result = rset->getInt(1);
    }
    std::cout << "id is : " << result << endl;

    m_conn->terminateStatement(m_readStmt);
    return result;

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODTTCFConfig::fetchNextId():  ") + e.getMessage()));
  }
}

void ODTTCFConfig::prepareWrite() noexcept(false) {
  this->checkConnection();
  int next_id = fetchNextId();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO ECAL_TTCF_CONFIGURATION (ttcf_configuration_id, ttcf_tag, "
        " rxbc0_delay, reg_30 , ttcf_configuration_file , ttcf_configuration ) "
        "VALUES (:1, :2, :3 , :4, :5, :6)");
    m_writeStmt->setInt(1, next_id);
    m_writeStmt->setString(2, getConfigTag());

    m_writeStmt->setInt(3, getRxBC0Delay());
    m_writeStmt->setInt(4, getReg30());

    m_writeStmt->setString(5, getTTCFConfigurationFile());

    oracle::occi::Clob clob(m_conn);
    clob.setEmpty();
    m_writeStmt->setClob(6, clob);
    m_writeStmt->executeUpdate();
    m_ID = next_id;

    m_conn->terminateStatement(m_writeStmt);
    std::cout << "inserted into CONFIGURATION with id=" << next_id << std::endl;

    // now we read and update it
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "SELECT ttcf_configuration FROM ECAL_TTCF_CONFIGURATION WHERE"
        " ttcf_configuration_id=:1 FOR UPDATE");

    std::cout << "updating the clob 0" << std::endl;

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODTTCFConfig::prepareWrite():  ") + e.getMessage()));
  }

  std::cout << "updating the clob 1 " << std::endl;
}

void ODTTCFConfig::writeDB() noexcept(false) {
  std::cout << "updating the clob 2" << std::endl;

  try {
    m_writeStmt->setInt(1, m_ID);
    ResultSet* rset = m_writeStmt->executeQuery();

    rset->next();

    oracle::occi::Clob clob = rset->getClob(1);
    cout << "Opening the clob in read write mode" << endl;
    populateClob(clob, getTTCFConfigurationFile(), m_size);
    int clobLength = clob.length();
    cout << "Length of the clob is: " << clobLength << endl;

    m_writeStmt->executeUpdate();
    m_writeStmt->closeResultSet(rset);

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODTTCFConfig::writeDB():  ") + e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODTTCFConfig::writeDB:  Failed to write"));
  }
}

void ODTTCFConfig::fetchData(ODTTCFConfig* result) noexcept(false) {
  this->checkConnection();
  result->clear();

  if (result->getId() == 0 && (result->getConfigTag().empty())) {
    throw(std::runtime_error("ODTTCFConfig::fetchData(): no Id defined for this ODTTCFConfig "));
  }

  try {
    m_readStmt->setSQL(
        "SELECT *   "
        "FROM ECAL_TTCF_CONFIGURATION  "
        " where (ttcf_configuration_id = :1 or ttcf_tag= :2) ");
    m_readStmt->setInt(1, result->getId());
    m_readStmt->setString(2, result->getConfigTag());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));
    result->setTTCFConfigurationFile(rset->getString(3));
    Clob clob = rset->getClob(4);
    cout << "Opening the clob in Read only mode" << endl;
    clob.open(OCCI_LOB_READONLY);
    int clobLength = clob.length();
    cout << "Length of the clob is: " << clobLength << endl;
    m_size = clobLength;
    unsigned char* buffer = readClob(clob, m_size);
    clob.close();
    result->setTTCFClob((unsigned char*)buffer);

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODTTCFConfig::fetchData():  ") + e.getMessage()));
  }
}

int ODTTCFConfig::fetchID() noexcept(false) {
  if (m_ID != 0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL(
        "SELECT ttcf_configuration_id FROM ecal_ttcf_configuration "
        "WHERE  ttcf_tag=:ttcf_tag ");

    stmt->setString(1, getConfigTag());

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODTTCFConfig::fetchID:  ") + e.getMessage()));
  }

  return m_ID;
}

void ODTTCFConfig::setParameters(const std::map<string, string>& my_keys_map) {
  // parses the result of the XML parser that is a map of
  // string string with variable name variable value

  for (std::map<std::string, std::string>::const_iterator ci = my_keys_map.begin(); ci != my_keys_map.end(); ci++) {
    if (ci->first == "TTCF_CONFIGURATION_ID")
      setConfigTag(ci->second);
    if (ci->first == "Configuration") {
      std::string fname = ci->second;
      string str3;
      size_t pos, pose;

      pos = fname.find('=');  // position of "live" in str
      pose = fname.size();    // position of "]" in str
      str3 = fname.substr(pos + 1, pose - pos - 2);

      cout << "fname=" << fname << " and reduced is: " << str3 << endl;
      setTTCFConfigurationFile(str3);

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

    } else if (ci->first == "RXBC0_DELAY") {
      setRxBC0Delay(atoi(ci->second.c_str()));
    } else if (ci->first == "REG_30") {
      setReg30(atoi(ci->second.c_str()));
    }
  }
}
