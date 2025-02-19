#include <stdexcept>
#include <cstdlib>
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
  m_config_tag="";
  m_ID=0;
  clear();

}

void ODLTSConfig::clear(){
  m_trg_type="";
  m_num=0;
  m_rate=0;
  m_delay=0;
}



ODLTSConfig::~ODLTSConfig()
{
}

void ODLTSConfig::setParameters(std::map<string,string> my_keys_map){
  
  // parses the result of the XML parser that is a map of 
  // string string with variable name variable value 
  
  for( std::map<std::string, std::string >::iterator ci=
	 my_keys_map.begin(); ci!=my_keys_map.end(); ci++ ) {

    if(ci->first==  "LTS_CONFIGURATION_ID") setConfigTag(ci->second);
    if(ci->first==  "NUM_OF_EVENTS") setNumberOfEvents(atoi(ci->second.c_str()) );
    if(ci->first==  "RATE") setRate(atoi(ci->second.c_str() ));
    if(ci->first==  "TRIGGER_TYPE") setTriggerType(ci->second );
    if(ci->first==  "TRIG_LOC_L1_DELAY") setTrigLocL1Delay(atoi(ci->second.c_str() ));
  }
  
}

int ODLTSConfig::fetchNextId()  throw(std::runtime_error) {

  int result=0;
  try {
    this->checkConnection();

    m_readStmt = m_conn->createStatement(); 
    m_readStmt->setSQL("select ecal_lts_config_sq.NextVal from dual");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next ()){
      result= rset->getInt(1);
    }
    m_conn->terminateStatement(m_readStmt);
    return result; 

  } catch (SQLException &e) {
    throw(std::runtime_error("ODLTSConfig::fetchNextId():  "+e.getMessage()));
  }

}


void ODLTSConfig::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();
  int next_id=fetchNextId();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_LTS_CONFIGURATION ( lts_configuration_id, lts_tag, "
			"trigger_type, num_of_events, rate, trig_loc_l1_delay ) "
			"VALUES (  "
			":1, :2, :3, :4 , :5, :6 )");
    m_writeStmt->setInt(1, next_id);
    m_ID=next_id;

  } catch (SQLException &e) {
    throw(std::runtime_error("ODLTSConfig::prepareWrite():  "+e.getMessage()));
  }
}



void ODLTSConfig::writeDB()
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  try {

    m_writeStmt->setString(2, this->getConfigTag());
    m_writeStmt->setString(3, this->getTriggerType());
    m_writeStmt->setInt(4, this->getNumberOfEvents());
    m_writeStmt->setInt(5, this->getRate());
    m_writeStmt->setInt(6, this->getTrigLocL1Delay());

    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(std::runtime_error("ODLTSConfig::writeDB():  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODLTSConfig::writeDB:  Failed to write"));
  }


}



void ODLTSConfig::fetchData(ODLTSConfig * result)
  throw(std::runtime_error)
{
  this->checkConnection();
  result->clear();
  if(result->getId()==0 && (result->getConfigTag()=="") ){
    throw(std::runtime_error("ODLTSConfig::fetchData(): no Id defined for this ODLTSConfig "));
  }

  try {

    m_readStmt->setSQL("SELECT * "
		       "FROM ECAL_LTS_CONFIGURATION  "
		       " where ( lts_configuration_id = :1 or lts_tag=:2 ) " );
    m_readStmt->setInt(1, result->getId());
    m_readStmt->setString(2, result->getConfigTag());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();
    // 1 is the id and 2 is the config tag
    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));

    result->setTriggerType(        rset->getString(3) );
    result->setNumberOfEvents(     rset->getInt(4) );
    result->setRate(               rset->getInt(5) );
    result->setTrigLocL1Delay(     rset->getInt(6) );
  

  } catch (SQLException &e) {
    throw(std::runtime_error("ODLTSConfig::fetchData():  "+e.getMessage()));
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
                 "WHERE  lts_tag=:lts_tag  " );

    stmt->setString(1, getConfigTag());

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("ODLTSConfig::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}
