#include <stdexcept>
#include <string>
#include <cstring>
#include <cstdlib>

#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODCCSConfig.h"

#include <limits>

#define MY_NULL 16777215  // numeric_limits<int>::quiet_NaN()
#define SET_INT(statement, paramNum, paramVal) \
  if (paramVal != MY_NULL) {                   \
    statement->setInt(paramNum, paramVal);     \
  } else {                                     \
    statement->setNull(paramNum, OCCINUMBER);  \
  }
#define SET_STRING(statement, paramNum, paramVal) \
  if (!paramVal.empty()) {                        \
    statement->setString(paramNum, paramVal);     \
  } else {                                        \
    statement->setNull(paramNum, OCCICHAR);       \
  }

using namespace std;
using namespace oracle::occi;

ODCCSConfig::ODCCSConfig() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;
  m_config_tag = "";
  m_ID = 0;
  clear();
}

void ODCCSConfig::clear() {
  m_daccal = MY_NULL;
  m_delay = MY_NULL;
  m_gain = "";
  m_memgain = "";
  m_offset_high = MY_NULL;
  m_offset_low = MY_NULL;
  m_offset_mid = MY_NULL;
  m_trg_mode = "";
  m_trg_filter = "";
  m_bgo = "";
  m_tts_mask = MY_NULL;
  m_daq = MY_NULL;
  m_trg = MY_NULL;
  m_bc0 = MY_NULL;
  m_bc0_delay = MY_NULL;
  m_te_delay = MY_NULL;
}

ODCCSConfig::~ODCCSConfig() {}

int ODCCSConfig::fetchNextId() noexcept(false) {
  int result = 0;
  try {
    this->checkConnection();

    m_readStmt = m_conn->createStatement();
    m_readStmt->setSQL("select ecal_CCS_config_sq.NextVal from dual");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next()) {
      result = rset->getInt(1);
    }
    m_conn->terminateStatement(m_readStmt);
    return result;

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODCCSConfig::fetchNextId():  ") + e.getMessage()));
  }
}

void ODCCSConfig::prepareWrite() noexcept(false) {
  this->checkConnection();
  int next_id = fetchNextId();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO ECAL_CCS_CONFIGURATION ( ccs_configuration_id, ccs_tag ,"
        " daccal, delay, gain, memgain, offset_high,offset_low,offset_mid, trg_mode, trg_filter, "
        " clock, BGO_SOURCE, TTS_MASK, DAQ_BCID_PRESET , TRIG_BCID_PRESET, BC0_COUNTER, BC0_DELAY, TE_DELAY ) "
        "VALUES (  "
        " :ccs_configuration_id, :ccs_tag,  :daccal, :delay, :gain, :memgain, :offset_high,:offset_low,"
        " :offset_mid, :trg_mode, :trg_filter, :clock, :BGO_SOURCE, :TTS_MASK, "
        " :DAQ_BCID_PRESET , :TRIG_BCID_PRESET, :BC0_COUNTER, :BC0_DELAY, :TE_DELAY) ");

    m_writeStmt->setInt(1, next_id);
    m_ID = next_id;

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODCCSConfig::prepareWrite():  ") + e.getMessage()));
  }
}

void ODCCSConfig::setParameters(const std::map<string, string>& my_keys_map) {
  // parses the result of the XML parser that is a map of
  // string string with variable name variable value

  for (std::map<std::string, std::string>::const_iterator ci = my_keys_map.begin(); ci != my_keys_map.end(); ci++) {
    if (ci->first == "CCS_CONFIGURATION_ID")
      setConfigTag(ci->second);
    else if (ci->first == "DACCAL")
      setDaccal(atoi(ci->second.c_str()));
    else if (ci->first == "GAIN")
      setGain(ci->second);
    else if (ci->first == "MEMGAIN")
      setMemGain(ci->second);
    else if (ci->first == "OFFSET_HIGH")
      setOffsetHigh(atoi(ci->second.c_str()));
    else if (ci->first == "OFFSET_LOW")
      setOffsetLow(atoi(ci->second.c_str()));
    else if (ci->first == "OFFSET_MID")
      setOffsetMid(atoi(ci->second.c_str()));
    else if (ci->first == "TRG_MODE")
      setTrgMode(ci->second);
    else if (ci->first == "TRG_FILTER")
      setTrgFilter(ci->second);
    else if (ci->first == "CLOCK")
      setClock(atoi(ci->second.c_str()));
    else if (ci->first == "BGO_SOURCE")
      setBGOSource(ci->second);
    else if (ci->first == "TTS_MASK")
      setTTSMask(atoi(ci->second.c_str()));
    else if (ci->first == "DAQ_BCID_PRESET")
      setDAQBCIDPreset(atoi(ci->second.c_str()));
    else if (ci->first == "TRIG_BCID_PRESET")
      setTrgBCIDPreset(atoi(ci->second.c_str()));
    else if (ci->first == "BC0_COUNTER")
      setBC0Counter(atoi(ci->second.c_str()));
    else if (ci->first == "BC0_DELAY")
      setBC0Delay(atoi(ci->second.c_str()));
    else if (ci->first == "TE_DELAY")
      setTEDelay(atoi(ci->second.c_str()));
    else if (ci->first == "DELAY")
      setDelay(atoi(ci->second.c_str()));
  }
}

void ODCCSConfig::writeDB() noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  try {
    // number 1 is the id
    // m_writeStmt->setString(2, this->getConfigTag());
    // m_writeStmt->setInt(3, this->getDaccal());
    // m_writeStmt->setInt(4, this->getDelay());
    // m_writeStmt->setString(5, this->getGain());
    // m_writeStmt->setString(6, this->getMemGain());
    // m_writeStmt->setInt(7, this->getOffsetHigh());
    // m_writeStmt->setInt(8, this->getOffsetLow());
    // m_writeStmt->setInt(9, this->getOffsetMid());
    // m_writeStmt->setString(10, this->getTrgMode() );
    // m_writeStmt->setString(11, this->getTrgFilter() );
    // m_writeStmt->setInt(  12, this->getClock() );
    // m_writeStmt->setString(13, this->getBGOSource() );
    // m_writeStmt->setInt(14, this->getTTSMask() );
    // m_writeStmt->setInt(15, this->getDAQBCIDPreset() );
    // m_writeStmt->setInt(16, this->getTrgBCIDPreset() );
    // m_writeStmt->setInt(17, this->getBC0Counter() );

    SET_STRING(m_writeStmt, 2, this->getConfigTag());
    SET_INT(m_writeStmt, 3, this->getDaccal());
    SET_INT(m_writeStmt, 4, this->getDelay());
    SET_STRING(m_writeStmt, 5, this->getGain());
    SET_STRING(m_writeStmt, 6, this->getMemGain());
    SET_INT(m_writeStmt, 7, this->getOffsetHigh());
    SET_INT(m_writeStmt, 8, this->getOffsetLow());
    SET_INT(m_writeStmt, 9, this->getOffsetMid());
    SET_STRING(m_writeStmt, 10, this->getTrgMode());
    SET_STRING(m_writeStmt, 11, this->getTrgFilter());
    SET_INT(m_writeStmt, 12, this->getClock());
    SET_STRING(m_writeStmt, 13, this->getBGOSource());
    SET_INT(m_writeStmt, 14, this->getTTSMask());
    SET_INT(m_writeStmt, 15, this->getDAQBCIDPreset());
    SET_INT(m_writeStmt, 16, this->getTrgBCIDPreset());
    SET_INT(m_writeStmt, 17, this->getBC0Counter());
    SET_INT(m_writeStmt, 18, this->getBC0Delay());
    SET_INT(m_writeStmt, 19, this->getTEDelay());

    m_writeStmt->executeUpdate();

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODCCSConfig::writeDB():  ") + e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODCCSConfig::writeDB:  Failed to write"));
  }
}

void ODCCSConfig::fetchData(ODCCSConfig* result) noexcept(false) {
  this->checkConnection();
  result->clear();
  if (result->getId() == 0 && (result->getConfigTag().empty())) {
    throw(std::runtime_error("ODCCSConfig::fetchData(): no Id defined for this ODCCSConfig "));
  }

  try {
    m_readStmt->setSQL(
        "SELECT * "
        "FROM ECAL_CCS_CONFIGURATION  "
        " where ( CCS_configuration_id = :1 or CCS_tag=:2 )");
    m_readStmt->setInt(1, result->getId());
    m_readStmt->setString(2, result->getConfigTag());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    // 1 is the id and 2 is the config tag

    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));

    result->setDaccal(rset->getInt(3));
    result->setDelay(rset->getInt(4));
    result->setGain(rset->getString(5));
    result->setMemGain(rset->getString(6));
    result->setOffsetHigh(rset->getInt(7));
    result->setOffsetLow(rset->getInt(8));
    result->setOffsetMid(rset->getInt(9));
    result->setTrgMode(rset->getString(10));
    result->setTrgFilter(rset->getString(11));
    result->setClock(rset->getInt(12));
    result->setBGOSource(rset->getString(13));
    result->setTTSMask(rset->getInt(14));
    result->setDAQBCIDPreset(rset->getInt(15));
    result->setTrgBCIDPreset(rset->getInt(16));
    result->setBC0Counter(rset->getInt(17));

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODCCSConfig::fetchData():  ") + e.getMessage()));
  }
}

int ODCCSConfig::fetchID() noexcept(false) {
  // Return from memory if available
  if (m_ID != 0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL(
        "SELECT ccs_configuration_id FROM ecal_ccs_configuration "
        "WHERE  ccs_tag=:ccs_tag ");

    stmt->setString(1, getConfigTag());

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODCCSConfig::fetchID:  ") + e.getMessage()));
  }

  return m_ID;
}
