#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODLTCConfig.h"

using namespace std;
using namespace oracle::occi;

ODLTCConfig::ODLTCConfig()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

   m_ID=0;
   m_dev=0;

}



ODLTCConfig::~ODLTCConfig()
{
}



void ODLTCConfig::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_LTC_CONFIGURATION (device_config_param_id ) "
			"VALUES ( :1) ");
  } catch (SQLException &e) {
    throw(runtime_error("ODLTCConfig::prepareWrite():  "+e.getMessage()));
  }
}



void ODLTCConfig::writeDB()
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  try {

    m_writeStmt->setInt(1, this->getDeviceConfigParamId());

    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(runtime_error("ODLTCConfig::writeDB():  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(runtime_error("ODLTCConfig::writeDB:  Failed to write"));
  }


}


void ODLTCConfig::clear(){
   m_dev=0;
}


void ODLTCConfig::fetchData(ODLTCConfig * result)
  throw(runtime_error)
{
  this->checkConnection();
  result->clear();
  if(result->getId()==0){
    throw(runtime_error("ODLTCConfig::fetchData(): no Id defined for this ODLTCConfig "));
  }

  try {

    m_readStmt->setSQL("SELECT d.device_config_param_id  "
		       "FROM ECAL_LTC_CONFIGURATION d "
		       " where ltc_configuration_id = :1 " );
    m_readStmt->setInt(1, result->getId());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setDeviceConfigParamId(       rset->getInt(1) );
 

  } catch (SQLException &e) {
    throw(runtime_error("ODLTCConfig::fetchData():  "+e.getMessage()));
  }
}

int ODLTCConfig::fetchID()    throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID!=0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT ltc_configuration_id FROM ecal_ltc_configuration "
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
    throw(runtime_error("ODLTCConfig::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}
