#include <string>
#include "occi.h"

#include "OnlineDB/EcalCondDB/interface/RunTypeDef.h"

using namespace std;
using namespace oracle::occi;

RunTypeDef::RunTypeDef()
{
  m_env = NULL;
  m_conn = NULL;
  m_ID = 0;
  m_runType = "";
  m_configTag = "";
  m_configVer = 0;
  m_desc = "";
}



RunTypeDef::~RunTypeDef()
{
}



string RunTypeDef::getRunType() const
{
  return m_runType;
}



void RunTypeDef::setRunType(string runtype)
{
  m_ID = 0;
  m_runType = runtype;
}



string RunTypeDef::getConfigTag() const
{
  return m_configTag;
}



void RunTypeDef::setConfigTag(string tag)
{
  m_ID = 0;
  m_configTag = tag;
}



int RunTypeDef::getConfigVersion() const
{
  return m_configVer;
}



void RunTypeDef::setConfigVersion(int ver)
{
  m_ID = 0;
  m_configVer = ver;
}



string RunTypeDef::getDescription() const
{
  return m_desc;
}


  
int RunTypeDef::fetchID()
  throw(std::runtime_error)
{
  // Return def from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();
  
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT def_id FROM run_type_def WHERE "
		 "run_type   = :1 AND "
		 "config_tag = :2 AND "
		 "config_ver = :3"
		 );
    stmt->setString(1, m_runType);
    stmt->setString(2, m_configTag);
    stmt->setInt(3, m_configVer);

    ResultSet* rset = stmt->executeQuery();
    
    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("RunTypeDef::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}



void RunTypeDef::setByID(int id) 
  throw(runtime_error)
{
  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("SELECT run_type, config_tag, config_ver, description FROM run_type_def WHERE def_id = :1");
    stmt->setInt(1, id);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      m_runType = rset->getString(1);
      m_configTag = rset->getString(2);
      m_configVer = rset->getInt(3);
      m_desc = rset->getString(4);
    } else {
      throw(runtime_error("RunTypeDef::setByID:  Given def_id is not in the database"));
    }
    
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
   throw(runtime_error("RunTypeDef::setByID:  "+e.getMessage()));
  }
}



void RunTypeDef::fetchAllDefs( std::vector<RunTypeDef>* fillVec) 
  throw(std::runtime_error)
{
  this->checkConnection();
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT def_id FROM run_type_def ORDER BY def_id");
    ResultSet* rset = stmt->executeQuery();
    
    RunTypeDef runTypeDef;
    runTypeDef.setConnection(m_env, m_conn);

    while(rset->next()) {
      runTypeDef.setByID( rset->getInt(1) );
      fillVec->push_back( runTypeDef );
    }
  } catch (SQLException &e) {
    throw(runtime_error("RunTypeDef::fetchAllDefs:  "+e.getMessage()));
  }
}
