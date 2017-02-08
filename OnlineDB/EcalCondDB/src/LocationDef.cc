#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"


#include "OnlineDB/EcalCondDB/interface/LocationDef.h"

using namespace std;
using namespace oracle::occi;

LocationDef::LocationDef()
{
  m_env = NULL;
  m_conn = NULL;
  m_ID = 0;
  m_loc = "";
}



LocationDef::~LocationDef()
{
}



string LocationDef::getLocation() const
{
  return m_loc;
}



void LocationDef::setLocation(string loc)
{
  if (loc != m_loc) {
    m_ID = 0;
    m_loc = loc;
  }
}


  
int LocationDef::fetchID()
  throw(std::runtime_error)
{
  // Return def from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();
  
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT def_id FROM location_def WHERE "
		 "location = :location");
    stmt->setString(1, m_loc);

    ResultSet* rset = stmt->executeQuery();
    
    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("LocationDef::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}



void LocationDef::setByID(int id) 
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("SELECT location FROM location_def WHERE def_id = :1");
    stmt->setInt(1, id);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      m_loc = rset->getString(1);
      m_ID = id;
    } else {
      throw(std::runtime_error("LocationDef::setByID:  Given def_id is not in the database"));
    }
    
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
   throw(std::runtime_error("LocationDef::setByID:  "+e.getMessage()));
  }
}



void LocationDef::fetchAllDefs( std::vector<LocationDef>* fillVec) 
  throw(std::runtime_error)
{
  this->checkConnection();
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT def_id FROM location_def ORDER BY def_id");
    ResultSet* rset = stmt->executeQuery();
    
    LocationDef locationDef;
    locationDef.setConnection(m_env, m_conn);

    while(rset->next()) {
      locationDef.setByID( rset->getInt(1) );
      fillVec->push_back( locationDef );
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("LocationDef::fetchAllDefs:  "+e.getMessage()));
  }
}
