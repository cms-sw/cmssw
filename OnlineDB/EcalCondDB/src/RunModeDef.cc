#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/RunModeDef.h"

using namespace std;
using namespace oracle::occi;

RunModeDef::RunModeDef()
{
  m_env = NULL;
  m_conn = NULL;
  m_ID = 0;
  m_runMode = "";

}



RunModeDef::~RunModeDef()
{
}



string RunModeDef::getRunMode() const
{
  return m_runMode;
}



void RunModeDef::setRunMode(string runmode)
{
  if (runmode != m_runMode) {
    m_ID = 0;
    m_runMode = runmode;
  }
}





  
int RunModeDef::fetchID()
  throw(std::runtime_error)
{
  // Return def from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();
  
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT def_id FROM ecal_run_mode_def WHERE "
		 "run_mode_string   = :1"
		 );
    stmt->setString(1, m_runMode);

    ResultSet* rset = stmt->executeQuery();
    
    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("RunModeDef::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}



void RunModeDef::setByID(int id) 
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("SELECT run_mode_string FROM ecal_run_mode_def WHERE def_id = :1");
    stmt->setInt(1, id);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      m_runMode = rset->getString(1);
    } else {
      throw(std::runtime_error("RunModeDef::setByID:  Given def_id is not in the database"));
    }
    
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
   throw(std::runtime_error("RunModeDef::setByID:  "+e.getMessage()));
  }
}



void RunModeDef::fetchAllDefs( std::vector<RunModeDef>* fillVec) 
  throw(std::runtime_error)
{
  this->checkConnection();
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT def_id FROM ecal_run_mode_def ORDER BY def_id");
    ResultSet* rset = stmt->executeQuery();
    
    RunModeDef runModeDef;
    runModeDef.setConnection(m_env, m_conn);

    while(rset->next()) {
      runModeDef.setByID( rset->getInt(1) );
      fillVec->push_back( runModeDef );
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("RunModeDef::fetchAllDefs:  "+e.getMessage()));
  }
}
