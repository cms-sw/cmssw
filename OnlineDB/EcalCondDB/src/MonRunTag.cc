#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"

using namespace std;
using namespace oracle::occi;

MonRunTag::MonRunTag()
{
  m_env = NULL;
  m_conn = NULL;
  m_ID = 0;
  m_genTag = "default";
  m_monVersionDef = MonVersionDef();
}



MonRunTag::~MonRunTag()
{
}



string MonRunTag::getGeneralTag() const
{
  return m_genTag;
}



void MonRunTag::setGeneralTag(string genTag)
{ 
  if (genTag != m_genTag) {
    m_ID = 0;
    m_genTag = genTag;
  }
}



MonVersionDef MonRunTag::getMonVersionDef() const
{
  return m_monVersionDef;
}


void MonRunTag::setMonVersionDef(MonVersionDef ver)
{
  if (ver != m_monVersionDef) {
    m_ID = 0;
    m_monVersionDef = ver;
  }
}



int MonRunTag::fetchID()
  throw(std::runtime_error)
{
  // Return tag from memory if available
  if (m_ID) {
    return m_ID;
  }
  
  this->checkConnection();

  // fetch parent IDs
  int verID;
  this->fetchParentIDs(&verID);

  // fetch this ID
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT tag_id FROM mon_run_tag WHERE "
		 "gen_tag    = :1 AND "
		 "mon_ver_id = :2");

    stmt->setString(1, m_genTag);
    stmt->setInt(2, verID);

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("MonRunTag::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}



void MonRunTag::setByID(int id) 
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("SELECT gen_tag, mon_ver_id FROM mon_run_tag WHERE tag_id = :1");
    stmt->setInt(1, id);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      m_genTag = rset->getString(1);
      int verID = rset->getInt(2);
      m_monVersionDef.setByID(verID);
      m_ID = id;
    } else {
      throw(std::runtime_error("MonRunTag::setByID:  Given tag_id is not in the database"));
    }

    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
   throw(std::runtime_error("MonRunTag::setByID:  "+e.getMessage()));
  }
}


int MonRunTag::writeDB()
  throw(std::runtime_error)
{
  // see if this data is already in the DB
  if (this->fetchID()) { 
     return m_ID; 
  }

  // check the connectioin
  this->checkConnection();

  // fetch parent IDs
  int verID;
  this->fetchParentIDs(&verID);

  // write new tag to the DB
  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("INSERT INTO mon_run_tag (tag_id, gen_tag, mon_ver_id) "
		 "VALUES (mon_run_tag_sq.NextVal, :1, :2)");
    stmt->setString(1, m_genTag);
    stmt->setInt(2, verID);

    stmt->executeUpdate();
    
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
   throw(std::runtime_error("MonRunTag::writeDB:  "+e.getMessage()));
  }

  // now get the tag_id
  if (!this->fetchID()) {
    throw(std::runtime_error("MonRunTag::writeDB:  Failed to write"));
  }

  return m_ID;
}



void MonRunTag::fetchAllTags( std::vector<MonRunTag>* fillVec)
  throw(std::runtime_error)
{
  this->checkConnection();
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT tag_id FROM mon_run_tag ORDER BY tag_id");
    ResultSet* rset = stmt->executeQuery();
    
    MonRunTag runtag;
    runtag.setConnection(m_env, m_conn);
    while(rset->next()) {
      runtag.setByID( rset->getInt(1) );
      fillVec->push_back( runtag );
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("MonRunTag::fetchAllTags:  "+e.getMessage()));
  }
}



void MonRunTag::fetchParentIDs(int* verID)
  throw(std::runtime_error)
{
  // get the monitoring version
  m_monVersionDef.setConnection(m_env, m_conn);
  *verID = m_monVersionDef.fetchID();

  if (! *verID) {
    throw(std::runtime_error("MonRunTag::writeDB:  Given monitoring version does not exist in DB")); 
  }
}
