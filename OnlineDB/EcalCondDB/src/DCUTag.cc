#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/DCUTag.h"

using namespace std;
using namespace oracle::occi;

DCUTag::DCUTag()
{
  m_env = NULL;
  m_conn = NULL;
  m_ID = 0;
  m_genTag = "default";
  m_locDef = LocationDef();
}



DCUTag::~DCUTag()
{
}



string DCUTag::getGeneralTag() const
{
  return m_genTag;
}

// User data methods



void DCUTag::setGeneralTag(string genTag)
{
  if (genTag != m_genTag) {
    m_ID = 0;
    m_genTag = genTag;
  }
}



LocationDef DCUTag::getLocationDef() const
{
  return m_locDef;
}



void DCUTag::setLocationDef(const LocationDef locDef)
{
  if (locDef != m_locDef) {
    m_ID = 0;
    m_locDef = locDef;
  }
}



int DCUTag::fetchID()
  throw(std::runtime_error)
{
  // Return tag from memory if available
  if (m_ID) {
    return m_ID;
  }
  
  this->checkConnection();

  // fetch the parent IDs
  int locID;
  this->fetchParentIDs(&locID);

  // fetch this ID
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT tag_id FROM dcu_tag WHERE "
		 "gen_tag     = :1 AND "
		 "location_id = :2");
    stmt->setString(1, m_genTag);
    stmt->setInt(2, locID);

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("DCUTag::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}



void DCUTag::setByID(int id) 
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("SELECT gen_tag, location_id FROM dcu_tag WHERE tag_id = :1");
    stmt->setInt(1, id);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      m_genTag = rset->getString(1);
      int locID = rset->getInt(2);

      m_locDef.setConnection(m_env, m_conn);
      m_locDef.setByID(locID);

      m_ID = id;
    } else {
      throw(std::runtime_error("DCUTag::setByID:  Given tag_id is not in the database"));
    }

    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
   throw(std::runtime_error("DCUTag::setByID:  "+e.getMessage()));
  }
}


int DCUTag::writeDB()
  throw(std::runtime_error)
{
  // see if this data is already in the DB
  if (this->fetchID()) { 
    return m_ID; 
  }

  // check the connectioin
  this->checkConnection();

  // fetch the parent IDs
  int locID;
  this->fetchParentIDs(&locID);

  // write new tag to the DB
  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("INSERT INTO dcu_tag (tag_id, gen_tag, location_id) "
		 "VALUES (dcu_tag_sq.NextVal, :1, :2)");
    stmt->setString(1, m_genTag);
    stmt->setInt(2, locID);

    stmt->executeUpdate();
    
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
   throw(std::runtime_error("DCUTag::writeDB:  "+e.getMessage()));
  }

  // now get the tag_id
  if (!this->fetchID()) {
    throw(std::runtime_error("DCUTag::writeDB:  Failed to write"));
  }

  return m_ID;
}



void DCUTag::fetchAllTags( std::vector<DCUTag>* fillVec)
  throw(std::runtime_error)
{
  this->checkConnection();
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT tag_id FROM dcu_tag ORDER BY tag_id");
    ResultSet* rset = stmt->executeQuery();
    
    DCUTag dcutag;
    dcutag.setConnection(m_env, m_conn);
    while(rset->next()) {
      dcutag.setByID( rset->getInt(1) );
      fillVec->push_back( dcutag );
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("DCUTag::fetchAllTags:  "+e.getMessage()));
  }
}



void DCUTag::fetchParentIDs(int* locID)
  throw(std::runtime_error)
{
  // get the location
  m_locDef.setConnection(m_env, m_conn);
  *locID = m_locDef.fetchID();

  if (! *locID) { 
    throw(std::runtime_error("DCUTag::writeDB:  Given location does not exist in DB")); 
  }
}
