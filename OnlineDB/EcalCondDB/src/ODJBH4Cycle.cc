#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODJBH4Cycle.h"

using namespace std;
using namespace oracle::occi;

ODJBH4Cycle::ODJBH4Cycle()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;
  //
  m_ID = 0;
  m_jbh4_config_id = 0;
}


ODJBH4Cycle::~ODJBH4Cycle()
{
}


void ODJBH4Cycle::prepareWrite()
  noexcept(false)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_JBH4_Cycle (cycle_id, jbh4_configuration_id ) "
		 "VALUES (:1, :2 )");
  } catch (SQLException &e) {
    throw(std::runtime_error("ODJBH4Cycle::prepareWrite():  "+e.getMessage()));
  }
}


void ODJBH4Cycle::writeDB()  noexcept(false)
{
  this->checkConnection();
  this->checkPrepare();

  try {

    m_writeStmt->setInt(1, this->getId());
    m_writeStmt->setInt(2, this->getJBH4ConfigurationID());

    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(std::runtime_error("ODJBH4Cycle::writeDB:  "+e.getMessage()));
  }

  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODJBH4Cycle::writeDB:  Failed to write"));
  }
  
 
}

void ODJBH4Cycle::clear(){
  m_jbh4_config_id=0;
}


int ODJBH4Cycle::fetchID()
  noexcept(false)
{
  // Return from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cycle_id, jbh4_configuration_id FROM ecal_jbh4_cycle "
		 "WHERE cycle_id = :1 ");
    stmt->setInt(1, m_ID);
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
      m_jbh4_config_id = rset->getInt(2);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("ODJBH4Cycle::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}



void ODJBH4Cycle::setByID(int id) 
  noexcept(false)
{
   this->checkConnection();


  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cycle_id, jbh4_configuration_id FROM ecal_jbh4_cycle "
		 "WHERE cycle_id = :1 ");
    stmt->setInt(1, id);
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
      m_jbh4_config_id = rset->getInt(2);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("ODJBH4Cycle::fetchID:  "+e.getMessage()));
  }
}



void ODJBH4Cycle::fetchData(ODJBH4Cycle * result)
  noexcept(false)
{
  this->checkConnection();
  result->clear();

  if(result->getId()==0){
    throw(std::runtime_error("ODJBH4Config::fetchData(): no Id defined for this ODJBH4Config "));
  }

  try {

    m_readStmt->setSQL("SELECT  jbh4_configuration_id FROM ecal_jbh4_cycle "
		 "WHERE cycle_id = :1 ");

    m_readStmt->setInt(1, result->getId());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setJBH4ConfigurationID(       rset->getInt(1) );

  } catch (SQLException &e) {
    throw(std::runtime_error("ODJBH4Cycle::fetchData():  "+e.getMessage()));
  }
}

 void ODJBH4Cycle::insertConfig()
  noexcept(false)
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

