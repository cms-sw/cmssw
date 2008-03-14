#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODLTSConfig.h"

using namespace std;
using namespace oracle::occi;

ODLTSConfig::ODLTSConfig()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;
  
  m_ID=0;
  clear();

}



ODLTSConfig::~ODLTSConfig()
{
}



void ODLTSConfig::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_LTS_CONFIGURATION ( "
			"trigger_type, num_of_events, rate, trig_loc_l1_delay ) "
			"VALUES (  "
			":1, :2, :3, :4 )");
  } catch (SQLException &e) {
    throw(runtime_error("ODLTSConfig::prepareWrite():  "+e.getMessage()));
  }
}



void ODLTSConfig::writeDB()
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  try {

    m_writeStmt->setString(1, this->getTriggerType());
    m_writeStmt->setInt(2, this->getNumberOfEvents());
    m_writeStmt->setInt(3, this->getRate());
    m_writeStmt->setInt(4, this->getTrigLocL1Delay());

    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(runtime_error("ODLTSConfig::writeDB():  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(runtime_error("ODLTSConfig::writeDB:  Failed to write"));
  }


}


void ODLTSConfig::clear(){
  m_trg_type="";
  m_num=0;
  m_rate=0;
  m_delay=0;
}


void ODLTSConfig::fetchData(ODLTSConfig * result)
  throw(runtime_error)
{
  this->checkConnection();
  result->clear();
  if(result->getId()==0){
    throw(runtime_error("ODLTSConfig::fetchData(): no Id defined for this ODLTSConfig "));
  }

  try {

    m_readStmt->setSQL("SELECT d.trigger_type, d.num_of_events, d.rate, d.trig_loc_l1_delay   "
		       "FROM ECAL_LTS_CONFIGURATION d "
		       " where lts_configuration_id = :1 " );
    m_readStmt->setInt(1, result->getId());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setTriggerType(  rset->getString(1) );
    result->setNumberOfEvents(        rset->getInt(2) );
    result->setRate(         rset->getInt(3) );
    result->setTrigLocL1Delay(      rset->getInt(4) );
  

  } catch (SQLException &e) {
    throw(runtime_error("ODLTSConfig::fetchData():  "+e.getMessage()));
  }
}

int ODLTSConfig::fetchID()    throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID!=0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT lts_configuration_id FROM ecal_lts_configuration "
                 "WHERE   trigger_type=:1 AND NUM_OF_EVENTS=:2 AND RATE=:3 AND TRIG_LOC_L1_DELAY=:4 " );

    stmt->setString(1, getTriggerType());
    stmt->setInt(2, getNumberOfEvents());
    stmt->setInt(3, getRate());
    stmt->setInt(4, getTrigLocL1Delay());

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("ODLTSConfig::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}
