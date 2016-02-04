#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODDCUConfig.h"

using namespace std;
using namespace oracle::occi;

ODDCUConfig::ODDCUConfig()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;
  m_config_tag="";
  m_ID=0;
  clear();

}

void ODDCUConfig::clear(){
}



ODDCUConfig::~ODDCUConfig()
{
}

void ODDCUConfig::setParameters(std::map<string,string> my_keys_map){
  
  // parses the result of the XML parser that is a map of 
  // string string with variable name variable value 
  
  for( std::map<std::string, std::string >::iterator ci=
	 my_keys_map.begin(); ci!=my_keys_map.end(); ci++ ) {

    if(ci->first==  "DCU_CONFIGURATION_ID") setConfigTag(ci->second);
  }
  
}

int ODDCUConfig::fetchNextId()  throw(std::runtime_error) {

  int result=0;
  try {
    this->checkConnection();

    m_readStmt = m_conn->createStatement(); 
    m_readStmt->setSQL("select ecal_dcu_config_sq.NextVal from dual");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next ()){
      result= rset->getInt(1);
    }
    m_conn->terminateStatement(m_readStmt);
    return result; 

  } catch (SQLException &e) {
    throw(std::runtime_error("ODDCUConfig::fetchNextId():  "+e.getMessage()));
  }

}


void ODDCUConfig::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();
  int next_id=fetchNextId();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_DCU_CONFIGURATION ( dcu_configuration_id, dcu_tag ) "
			"VALUES (  "
			":1, :2 )");
    m_writeStmt->setInt(1, next_id);
    m_ID=next_id;

  } catch (SQLException &e) {
    throw(std::runtime_error("ODDCUConfig::prepareWrite():  "+e.getMessage()));
  }
}



void ODDCUConfig::writeDB()
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  try {

    m_writeStmt->setString(2, this->getConfigTag());

    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(std::runtime_error("ODDCUConfig::writeDB():  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODDCUConfig::writeDB:  Failed to write"));
  }


}



void ODDCUConfig::fetchData(ODDCUConfig * result)
  throw(std::runtime_error)
{
  this->checkConnection();
  result->clear();
  if(result->getId()==0 && (result->getConfigTag()=="") ){
    throw(std::runtime_error("ODDCUConfig::fetchData(): no Id defined for this ODDCUConfig "));
  }

  try {

    m_readStmt->setSQL("SELECT * "
		       "FROM ECAL_DCU_CONFIGURATION  "
		       " where ( dcu_configuration_id = :1 or dcu_tag=:2 ) " );
    m_readStmt->setInt(1, result->getId());
    m_readStmt->setString(2, result->getConfigTag());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();
    // 1 is the id and 2 is the config tag
    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));


  } catch (SQLException &e) {
    throw(std::runtime_error("ODDCUConfig::fetchData():  "+e.getMessage()));
  }
}

int ODDCUConfig::fetchID()    throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID!=0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT dcu_configuration_id FROM ecal_dcu_configuration "
                 "WHERE  dcu_tag=:dcu_tag  " );

    stmt->setString(1, getConfigTag());

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("ODDCUConfig::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}
