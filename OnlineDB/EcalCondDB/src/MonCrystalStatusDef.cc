#include <string>
#include "occi.h"

#include "OnlineDB/EcalCondDB/interface/MonCrystalStatusDef.h"


using namespace std;
using namespace oracle::occi;

MonCrystalStatusDef::MonCrystalStatusDef()
{
  m_env = NULL;
  m_conn = NULL;
  m_ID = 0;
  m_shortDesc = "";
  m_longDesc = "";

}



MonCrystalStatusDef::~MonCrystalStatusDef()
{
}



string MonCrystalStatusDef::getShortDesc() const
{
  return m_shortDesc;
}



void MonCrystalStatusDef::setShortDesc(string desc)
{
  m_ID = 0;
  m_shortDesc = desc;
}



string MonCrystalStatusDef::getLongDesc() const
{
  return m_longDesc;
}


  
int MonCrystalStatusDef::fetchID()
  throw(std::runtime_error)
{
  // Return def from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();
  
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT def_id FROM mon_crystal_status_def WHERE "
		 "short_desc = :1");
    stmt->setString(1, m_shortDesc);

    ResultSet* rset = stmt->executeQuery();
    
    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("MonCrystalStatusDef::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}



void MonCrystalStatusDef::setByID(int id) 
  throw(runtime_error)
{
  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("SELECT short_desc, long_desc FROM mon_crystal_status_def WHERE def_id = :1");
    stmt->setInt(1, id);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      m_shortDesc = rset->getString(1);
      m_longDesc = rset->getString(2);
      m_ID = id;
    } else {
      throw(runtime_error("MonCrystalStatusDef::setByID:  Given def_id is not in the database"));
    }
    
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
   throw(runtime_error("MonCrystalStatusDef::setByID:  "+e.getMessage()));
  }
}



void MonCrystalStatusDef::fetchAllDefs( std::vector<MonCrystalStatusDef>* fillVec) 
  throw(std::runtime_error)
{
  this->checkConnection();
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT def_id FROM mon_crystal_status_def ORDER BY def_id");
    ResultSet* rset = stmt->executeQuery();
    
    MonCrystalStatusDef def;
    def.setConnection(m_env, m_conn);

    while(rset->next()) {
      def.setByID( rset->getInt(1) );
      fillVec->push_back( def );
    }
  } catch (SQLException &e) {
    throw(runtime_error("MonCrystalStatusDef::fetchAllDefs:  "+e.getMessage()));
  }
}
