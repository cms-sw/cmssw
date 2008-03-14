#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODTCCConfig.h"

using namespace std;
using namespace oracle::occi;

ODTCCConfig::ODTCCConfig()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

   m_ID=0;
   m_dev=0;

}



ODTCCConfig::~ODTCCConfig()
{
}



void ODTCCConfig::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_TCC_CONFIGURATION (device_config_param_id ) "
			"VALUES ( :1) ");
  } catch (SQLException &e) {
    throw(runtime_error("ODTCCConfig::prepareWrite():  "+e.getMessage()));
  }
}



void ODTCCConfig::writeDB()
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  try {

    m_writeStmt->setInt(1, this->getDeviceConfigParamId());

    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(runtime_error("ODTCCConfig::writeDB():  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(runtime_error("ODTCCConfig::writeDB:  Failed to write"));
  }


}


void ODTCCConfig::clear(){
   m_dev=0;
}


void ODTCCConfig::fetchData(ODTCCConfig * result)
  throw(runtime_error)
{
  this->checkConnection();
  result->clear();
  if(result->getId()==0){
    throw(runtime_error("ODTCCConfig::fetchData(): no Id defined for this ODTCCConfig "));
  }

  try {

    m_readStmt->setSQL("SELECT d.device_config_param_id  "
		       "FROM ECAL_TCC_CONFIGURATION d "
		       " where tcc_configuration_id = :1 " );
    m_readStmt->setInt(1, result->getId());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setDeviceConfigParamId(       rset->getInt(1) );
 

  } catch (SQLException &e) {
    throw(runtime_error("ODTCCConfig::fetchData():  "+e.getMessage()));
  }
}

int ODTCCConfig::fetchID()    throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID!=0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT tcc_configuration_id FROM ecal_tcc_configuration "
                 "WHERE   device_config_param_id =:1 " );

    stmt->setInt(1, getDeviceConfigParamId());

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("ODTCCConfig::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}
