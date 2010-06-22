#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"

using namespace std;
using namespace oracle::occi;

LMFRunTag::LMFRunTag()
{
  m_env = NULL;
  m_conn = NULL;
  m_ID = 0;
  m_genTag = "default";
}



LMFRunTag::~LMFRunTag()
{
}



string LMFRunTag::getGeneralTag() const
{
  return m_genTag;
}



void LMFRunTag::setGeneralTag(string genTag)
{
  if (genTag != m_genTag) {
    m_ID = 0;
    m_genTag = genTag;
  }
}



int LMFRunTag::fetchID()
  throw(runtime_error)
{
  // Return tag from memory if available
  if (m_ID) {
    return m_ID;
  }
  
  this->checkConnection();

  // fetch this ID
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT tag_id FROM lmf_run_tag WHERE "
		 "gen_tag    = :1");

    stmt->setString(1, m_genTag);

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("LMFRunTag::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}



void LMFRunTag::setByID(int id) 
  throw(runtime_error)
{
  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("SELECT gen_tag FROM lmf_run_tag WHERE tag_id = :1");
    stmt->setInt(1, id);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      m_genTag = rset->getString(1);
      m_ID = id;
    } else {
      throw(runtime_error("LMFRunTag::setByID:  Given tag_id is not in the database"));
    }

    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
   throw(runtime_error("LMFRunTag::setByID:  "+e.getMessage()));
  }
}


int LMFRunTag::writeDB()
  throw(runtime_error)
{
  // see if this data is already in the DB
  if (this->fetchID()) { 
     return m_ID; 
  }

  // check the connectioin
  this->checkConnection();

  // write new tag to the DB
  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("INSERT INTO lmf_run_tag (tag_id, gen_tag) "
		 "VALUES (lmf_run_tag_sq.NextVal, :1)");
    stmt->setString(1, m_genTag);

    stmt->executeUpdate();
    
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
   throw(runtime_error("LMFRunTag::writeDB:  "+e.getMessage()));
  }

  // now get the tag_id
  if (!this->fetchID()) {
    throw(runtime_error("LMFRunTag::writeDB:  Failed to write"));
  }

  return m_ID;
}



void LMFRunTag::fetchAllTags( vector<LMFRunTag>* fillVec)
  throw(runtime_error)
{
  this->checkConnection();
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT tag_id FROM lmf_run_tag ORDER BY tag_id");
    ResultSet* rset = stmt->executeQuery();
    
    LMFRunTag runtag;
    runtag.setConnection(m_env, m_conn);
    while(rset->next()) {
      runtag.setByID( rset->getInt(1) );
      fillVec->push_back( runtag );
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("LMFRunTag::fetchAllTags:  "+e.getMessage()));
  }
}
