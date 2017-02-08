#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODScanCycle.h"

using namespace std;
using namespace oracle::occi;

ODScanCycle::ODScanCycle()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;
  //
  m_ID = 0;
  m_scan_config_id = 0;
}


ODScanCycle::~ODScanCycle()
{
}


void ODScanCycle::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_Scan_Cycle (cycle_id, scan_id ) "
		 "VALUES (:1, :2 )");
  } catch (SQLException &e) {
    throw(std::runtime_error("ODScanCycle::prepareWrite():  "+e.getMessage()));
  }
}


void ODScanCycle::writeDB()  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  try {

    m_writeStmt->setInt(1, this->getId());
    m_writeStmt->setInt(2, this->getScanConfigurationID());

    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(std::runtime_error("ODScanCycle::writeDB:  "+e.getMessage()));
  }

  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODScanCycle::writeDB:  Failed to write"));
  }
  
 
}

void ODScanCycle::clear(){
  m_scan_config_id=0;
}


int ODScanCycle::fetchID()
  throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cycle_id, scan_id FROM ecal_scan_cycle "
		 "WHERE cycle_id = :1 ");
    stmt->setInt(1, m_ID);
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
      m_scan_config_id = rset->getInt(2);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("ODScanCycle::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}



void ODScanCycle::setByID(int id) 
  throw(std::runtime_error)
{
   this->checkConnection();


  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cycle_id, scan_configuration_id FROM ecal_scan_cycle "
		 "WHERE cycle_id = :1 ");
    stmt->setInt(1, id);
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
      m_scan_config_id = rset->getInt(2);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("ODScanCycle::fetchID:  "+e.getMessage()));
  }
}



void ODScanCycle::fetchData(ODScanCycle * result)
  throw(std::runtime_error)
{
  this->checkConnection();
  result->clear();

  if(result->getId()==0){
    throw(std::runtime_error("ODScanConfig::fetchData(): no Id defined for this ODScanConfig "));
  }

  try {

    m_readStmt->setSQL("SELECT  scan_configuration_id FROM ecal_scan_cycle "
		 "WHERE cycle_id = :1 ");

    m_readStmt->setInt(1, result->getId());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setScanConfigurationID(       rset->getInt(1) );

  } catch (SQLException &e) {
    throw(std::runtime_error("ODScanCycle::fetchData():  "+e.getMessage()));
  }
}

void ODScanCycle::insertConfig()
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
