#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODDCCCycle.h"

using namespace std;
using namespace oracle::occi;

ODDCCCycle::ODDCCCycle()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;
  //
  m_ID = 0;
  m_dcc_config_id = 0;
}


ODDCCCycle::~ODDCCCycle()
{
}


void ODDCCCycle::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_DCC_Cycle (cycle_id, dcc_configuration_id ) "
		 "VALUES (:1, :2 )");
  } catch (SQLException &e) {
    throw(std::runtime_error("ODDCCCycle::prepareWrite():  "+e.getMessage()));
  }
}


void ODDCCCycle::writeDB()  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  try {

    m_writeStmt->setInt(1, this->getId());
    m_writeStmt->setInt(2, this->getDCCConfigurationID());

    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(std::runtime_error("ODDCCCycle::writeDB:  "+e.getMessage()));
  }

  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODDCCCycle::writeDB:  Failed to write"));
  }
  
 
}

void ODDCCCycle::clear(){
  m_dcc_config_id=0;
}


int ODDCCCycle::fetchID()
  throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cycle_id, dcc_configuration_id FROM ecal_dcc_cycle "
		 "WHERE cycle_id = :1 ");
    stmt->setInt(1, m_ID);
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
      m_dcc_config_id = rset->getInt(2);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("ODDCCCycle::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}



void ODDCCCycle::setByID(int id) 
  throw(std::runtime_error)
{
   this->checkConnection();


  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cycle_id, dcc_configuration_id FROM ecal_dcc_cycle "
		 "WHERE cycle_id = :1 ");
    stmt->setInt(1, id);
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
      m_dcc_config_id = rset->getInt(2);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("ODDCCCycle::fetchID:  "+e.getMessage()));
  }
}



void ODDCCCycle::fetchData(ODDCCCycle * result)
  throw(std::runtime_error)
{
  this->checkConnection();
  result->clear();

  if(result->getId()==0){
    throw(std::runtime_error("ODDCCConfig::fetchData(): no Id defined for this ODDCCConfig "));
  }

  try {

    m_readStmt->setSQL("SELECT  dcc_configuration_id FROM ecal_dcc_cycle "
		 "WHERE cycle_id = :1 ");

    m_readStmt->setInt(1, result->getId());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setDCCConfigurationID(       rset->getInt(1) );

  } catch (SQLException &e) {
    throw(std::runtime_error("ODDCCCycle::fetchData():  "+e.getMessage()));
  }
}

void ODDCCCycle::insertConfig()
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

