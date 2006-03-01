#include <string>
#include "occi.h"

#include "OnlineDB/EcalCondDB/interface/MonPNStatusDef.h"


using namespace std;
using namespace oracle::occi;

MonPNStatusDef::MonPNStatusDef()
{
  m_env = NULL;
  m_conn = NULL;
  m_ID = 0;
  m_shortDesc = "";
  m_longDesc = "";

}



MonPNStatusDef::~MonPNStatusDef()
{
}



string MonPNStatusDef::getShortDesc() const
{
  return m_shortDesc;
}



void MonPNStatusDef::setShortDesc(string desc)
{
  m_ID = 0;
  m_shortDesc = desc;
}



string MonPNStatusDef::getLongDesc() const
{
  return m_longDesc;
}


  
int MonPNStatusDef::fetchID()
  throw(std::runtime_error)
{
  // Return def from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();
  
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT def_id FROM mon_pn_status_def WHERE "
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
    throw(runtime_error("MonPNStatusDef::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}



void MonPNStatusDef::setByID(int id) 
  throw(runtime_error)
{
  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("SELECT short_desc, long_desc FROM mon_pn_status_def WHERE def_id = :1");
    stmt->setInt(1, id);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      m_shortDesc = rset->getString(1);
      m_longDesc = rset->getString(2);
      m_ID = id;
    } else {
      throw(runtime_error("MonPNStatusDef::setByID:  Given def_id is not in the database"));
    }
    
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
   throw(runtime_error("MonPNStatusDef::setByID:  "+e.getMessage()));
  }
}



void MonPNStatusDef::fetchAllDefs( std::vector<MonPNStatusDef>* fillVec) 
  throw(std::runtime_error)
{
  this->checkConnection();
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT def_id FROM mon_pn_status_def ORDER BY def_id");
    ResultSet* rset = stmt->executeQuery();
    
    MonPNStatusDef def;
    def.setConnection(m_env, m_conn);

    while(rset->next()) {
      def.setByID( rset->getInt(1) );
      fillVec->push_back( def );
    }
  } catch (SQLException &e) {
    throw(runtime_error("MonPNStatusDef::fetchAllDefs:  "+e.getMessage()));
  }
}
