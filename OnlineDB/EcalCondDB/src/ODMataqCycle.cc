#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODMataqCycle.h"

using namespace std;
using namespace oracle::occi;

ODMataqCycle::ODMataqCycle()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;
  //
  m_ID = 0;
  m_mataq_config_id = 0;
}


ODMataqCycle::~ODMataqCycle()
{
}


void ODMataqCycle::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_Matacq_Cycle (cycle_id, matacq_configuration_id ) "
		 "VALUES (:1, :2 )");
  } catch (SQLException &e) {
    throw(runtime_error("ODMataqCycle::prepareWrite():  "+e.getMessage()));
  }
}


void ODMataqCycle::writeDB()  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  try {

    m_writeStmt->setInt(1, this->getId());
    m_writeStmt->setInt(2, this->getMataqConfigurationID());

    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(runtime_error("ODMataqCycle::writeDB:  "+e.getMessage()));
  }

  // Now get the ID
  if (!this->fetchID()) {
    throw(runtime_error("ODMataqCycle::writeDB:  Failed to write"));
  }
  
 
}

void ODMataqCycle::clear(){
  m_mataq_config_id=0;
}


int ODMataqCycle::fetchID()
  throw(runtime_error)
{
  // Return from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cycle_id, matacq_configuration_id FROM ecal_matacq_cycle "
		 "WHERE cycle_id = :1 ");
    stmt->setInt(1, m_ID);
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
      m_mataq_config_id = rset->getInt(2);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("ODMataqCycle::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}



void ODMataqCycle::setByID(int id) 
  throw(std::runtime_error)
{
   this->checkConnection();


  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cycle_id, matacq_configuration_id FROM ecal_matacq_cycle "
		 "WHERE cycle_id = :1 ");
    stmt->setInt(1, id);
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
      m_mataq_config_id = rset->getInt(2);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("ODMataqCycle::fetchID:  "+e.getMessage()));
  }
}



void ODMataqCycle::fetchData(ODMataqCycle * result)
  throw(runtime_error)
{
  this->checkConnection();
  result->clear();

  if(result->getId()==0){
    throw(runtime_error("ODMataqConfig::fetchData(): no Id defined for this ODMataqConfig "));
  }

  try {

    m_readStmt->setSQL("SELECT  matacq_configuration_id FROM ecal_matacq_cycle "
		 "WHERE cycle_id = :1 ");

    m_readStmt->setInt(1, result->getId());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setMataqConfigurationID(       rset->getInt(1) );

  } catch (SQLException &e) {
    throw(runtime_error("ODMataqCycle::fetchData():  "+e.getMessage()));
  }
}

 void ODMataqCycle::insertConfig()
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

