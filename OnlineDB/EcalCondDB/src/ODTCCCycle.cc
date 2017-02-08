#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODTCCCycle.h"

using namespace std;
using namespace oracle::occi;

ODTCCCycle::ODTCCCycle()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;
  //
  m_ID = 0;
  m_tcc_config_id = 0;
}


ODTCCCycle::~ODTCCCycle()
{
}


void ODTCCCycle::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_TCC_Cycle (cycle_id, tcc_configuration_id ) "
		 "VALUES (:1, :2 )");
  } catch (SQLException &e) {
    throw(std::runtime_error("ODTCCCycle::prepareWrite():  "+e.getMessage()));
  }
}


void ODTCCCycle::writeDB()  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  try {

    m_writeStmt->setInt(1, this->getId());
    m_writeStmt->setInt(2, this->getTCCConfigurationID());

    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(std::runtime_error("ODTCCCycle::writeDB:  "+e.getMessage()));
  }

  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODTCCCycle::writeDB:  Failed to write"));
  }
  
 
}

void ODTCCCycle::clear(){
  m_tcc_config_id=0;
}


int ODTCCCycle::fetchID()
  throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cycle_id, tcc_configuration_id FROM ecal_tcc_cycle "
		 "WHERE cycle_id = :1 ");
    stmt->setInt(1, m_ID);
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
      m_tcc_config_id = rset->getInt(2);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("ODTCCCycle::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}



void ODTCCCycle::setByID(int id) 
  throw(std::runtime_error)
{
   this->checkConnection();


  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cycle_id, tcc_configuration_id FROM ecal_tcc_cycle "
		 "WHERE cycle_id = :1 ");
    stmt->setInt(1, id);
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
      m_tcc_config_id = rset->getInt(2);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("ODTCCCycle::fetchID:  "+e.getMessage()));
  }
}



void ODTCCCycle::fetchData(ODTCCCycle * result)
  throw(std::runtime_error)
{
  this->checkConnection();
  result->clear();

  if(result->getId()==0){
    throw(std::runtime_error("ODTCCConfig::fetchData(): no Id defined for this ODTCCConfig "));
  }

  try {

    m_readStmt->setSQL("SELECT  tcc_configuration_id FROM ecal_tcc_cycle "
		 "WHERE cycle_id = :1 ");

    m_readStmt->setInt(1, result->getId());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setTCCConfigurationID(       rset->getInt(1) );

  } catch (SQLException &e) {
    throw(std::runtime_error("ODTCCCycle::fetchData():  "+e.getMessage()));
  }
}

void ODTCCCycle::insertConfig()
  throw(std::runtime_error)
{
  try {

    prepareWrite();
    writeDB();
    m_conn->commit();
    terminateWriteStatement();
  } catch (std::runtime_error &e) {
    m_conn->rollback();
    throw(e);
  } catch (...) {
    m_conn->rollback();
    throw(std::runtime_error("EcalCondDBInterface::insertDataSet:  Unknown exception caught"));
  }
}
