#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODCCSConfig.h"

using namespace std;
using namespace oracle::occi;

ODCCSConfig::ODCCSConfig()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;
  m_config_tag="";
   m_ID=0;
   clear();   
}


void ODCCSConfig::clear(){
   m_daccal=0;
   m_delay=0;
   m_gain=0;
   m_memgain=0;
   m_offset_high=0;
   m_offset_low=0;
   m_offset_mid=0;
   m_trg_mode="";
   m_trg_filter="";
   m_bgo="";
   m_tts_mask=0;
   m_daq=0;
   m_trg=0;
   m_bc0=0;
}



ODCCSConfig::~ODCCSConfig()
{
}



int ODCCSConfig::fetchNextId()  throw(std::runtime_error) {

  int result=0;
  try {
    this->checkConnection();

    m_readStmt = m_conn->createStatement(); 
    m_readStmt->setSQL("select ecal_CCS_config_sq.NextVal from dual");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next ()){
      result= rset->getInt(1);
    }
    m_conn->terminateStatement(m_readStmt);
    return result; 

  } catch (SQLException &e) {
    throw(runtime_error("ODCCSConfig::fetchNextId():  "+e.getMessage()));
  }

}

void ODCCSConfig::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();
  int next_id=fetchNextId();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_CCS_CONFIGURATION ( ccs_configuration_id, ccs_tag ,"
			" daccal, delay, gain, memgain, offset_high,offset_low,offset_mid, trg_mode, trg_filter, "
			" clock, BGO_SOURCE, TTS_MASK, DAQ_BCID_PRESET , TRIG_BCID_PRESET, BC0_COUNTER ) "
			"VALUES (  "
			" :ccs_configuration_id, :ccs_tag,  :daccal, :delay, :gain, :memgain, :offset_high,:offset_low,"
			" :offset_mid, :trg_mode, :trg_filter, :clock, :BGO_SOURCE, :TTS_MASK, "
			" :DAQ_BCID_PRESET , :TRIG_BCID_PRESET, :BC0_COUNTER) ");

    m_writeStmt->setInt(1, next_id);
    m_ID=next_id;

  } catch (SQLException &e) {
    throw(runtime_error("ODCCSConfig::prepareWrite():  "+e.getMessage()));
  }

}



void ODCCSConfig::writeDB()
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  try {

    // number 1 is the id 
    m_writeStmt->setString(2, this->getConfigTag());
    m_writeStmt->setInt(3, this->getDaccal());
    m_writeStmt->setInt(4, this->getDelay());
    m_writeStmt->setInt(5, this->getGain());
    m_writeStmt->setInt(6, this->getMemGain());
    m_writeStmt->setInt(7, this->getOffsetHigh());
    m_writeStmt->setInt(8, this->getOffsetLow());
    m_writeStmt->setInt(9, this->getOffsetMid());
    m_writeStmt->setString(10, this->getTrgMode() );
    m_writeStmt->setString(11, this->getTrgFilter() );
    m_writeStmt->setInt(  12, this->getClock() );
    m_writeStmt->setString(13, this->getBGOSource() );
    m_writeStmt->setInt(14, this->getTTSMask() );
    m_writeStmt->setInt(15, this->getDAQBCIDPreset() );
    m_writeStmt->setInt(16, this->getTrgBCIDPreset() );
    m_writeStmt->setInt(17, this->getBC0Counter() );

    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(runtime_error("ODCCSConfig::writeDB():  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(runtime_error("ODCCSConfig::writeDB:  Failed to write"));
  }


}


void ODCCSConfig::fetchData(ODCCSConfig * result)
  throw(runtime_error)
{
  this->checkConnection();
  result->clear();
  if(result->getId()==0 && (result->getConfigTag()=="") ){
    throw(runtime_error("ODCCSConfig::fetchData(): no Id defined for this ODCCSConfig "));
  }

  try {

    m_readStmt->setSQL("SELECT * "
                       "FROM ECAL_CCS_CONFIGURATION  "
                       " where ( CCS_configuration_id = :1 or CCS_tag=:2 )" );
    m_readStmt->setInt(1, result->getId());
    m_readStmt->setString(2, result->getConfigTag());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    // 1 is the id and 2 is the config tag

    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));

    result->setDaccal(       rset->getInt(3) );
    result->setDelay(        rset->getInt(4) );
    result->setGain(         rset->getInt(5) );
    result->setMemGain(      rset->getInt(6) );
    result->setOffsetHigh(   rset->getInt(7) );
    result->setOffsetLow(    rset->getInt(8) );
    result->setOffsetMid(    rset->getInt(9) );
    result->setTrgMode(      rset->getString(10) );
    result->setTrgFilter(    rset->getString(11) );
    result->setClock(        rset->getInt(12) );
    result->setBGOSource(      rset->getString(13) );
    result->setTTSMask(        rset->getInt(14) );
    result->setDAQBCIDPreset(        rset->getInt(15) );
    result->setTrgBCIDPreset(        rset->getInt(16) );
    result->setBC0Counter(        rset->getInt(17) );


  } catch (SQLException &e) {
    throw(runtime_error("ODCCSConfig::fetchData():  "+e.getMessage()));
  }
}

int ODCCSConfig::fetchID()    throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID!=0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT ccs_configuration_id FROM ecal_ccs_configuration "
                 "WHERE  ccs_tag=:ccs_tag "
		 );

    stmt->setString(1, getConfigTag() );

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("ODCCSConfig::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}
