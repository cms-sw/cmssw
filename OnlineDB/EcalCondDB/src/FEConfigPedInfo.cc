#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigPedInfo.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;



FEConfigPedInfo::FEConfigPedInfo()
{
  m_conn = NULL;
  m_ID = 0;
  //
  m_iov_id =0 ;
  m_tag ="";

}



FEConfigPedInfo::~FEConfigPedInfo(){}


void FEConfigPedInfo::setID(int id){ m_ID = id;  }
int FEConfigPedInfo::getID(){ return m_ID ;  }

void FEConfigPedInfo::setIOVId(int iov_id){ m_iov_id = iov_id;  }
int FEConfigPedInfo::getIOVId()const {return m_iov_id;  }


Tm FEConfigPedInfo::getDBTime() const{  return m_db_time;}

void FEConfigPedInfo::setTag(std::string x) {
  if (x != m_tag) {
    m_ID = 0;
    m_tag = x;
  }
}

std::string FEConfigPedInfo::getTag() const{  return m_tag;}

//    
int FEConfigPedInfo::fetchID()
  throw(runtime_error)
{
  // Return from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();


  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT ped_conf_id FROM fe_config_ped_info "
		 "WHERE iov_id = :iov_id  ");
    stmt->setInt(1, m_iov_id);
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("FEConfigPedInfo::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}

//
int FEConfigPedInfo::fetchIDLast()
  throw(runtime_error)
{

  this->checkConnection();


  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT max( ped_conf_id) FROM fe_config_ped_info "	);
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("FEConfigPedInfo::fetchIDLast:  "+e.getMessage()));
  }

  return m_ID;
}

//
int FEConfigPedInfo::fetchIDFromTag()
  throw(runtime_error)
{
  // selects tha most recent config id with a given tag 

  this->checkConnection();

  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT max(ped_conf_id) FROM fe_config_ped_info "
		 "WHERE tag = :tag  ");
    stmt->setString(1, m_tag);
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("FEConfigPedInfo::fetchIDFromTag:  "+e.getMessage()));
  }
  return m_ID;
}


void FEConfigPedInfo::setByID(int id) 
  throw(std::runtime_error)
{
   this->checkConnection();

   DateHandler dh(m_env, m_conn);

   try {
     Statement* stmt = m_conn->createStatement();

     stmt->setSQL("SELECT iov_id, db_timestamp, tag FROM fe_config_ped_info WHERE ped_conf_id = :1");
     stmt->setInt(1, id);
     
     ResultSet* rset = stmt->executeQuery();
     if (rset->next()) {
       m_iov_id = rset->getInt(1);
       Date dbdate = rset->getDate(2);
       m_db_time = dh.dateToTm( dbdate );
       m_tag=rset->getString(3);
       m_ID = id;

     } else {
       throw(runtime_error("FEConfigPedInfo::setByID:  Given config_id is not in the database"));
     }
     
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(runtime_error("FEConfigPedInfo::setByID:  "+e.getMessage()));
   }
}



int FEConfigPedInfo::writeDB()
  throw(runtime_error)
{
  this->checkConnection();

  // Check if this IOV has already been written
  if (this->fetchID()) {
    return m_ID;
  }
  
  // Validate the data, use infinity-till convention
  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();
    
    stmt->setSQL("INSERT INTO FE_CONFIG_PED_INFO (ped_conf_id, iov_id, tag ) "
		 "VALUES (FE_CONFIG_PED_SQ.NextVal, :1, :2 )");
    stmt->setInt(1, m_iov_id);
    stmt->setString(2, m_tag );

    stmt->executeUpdate();

    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("FEConfigPedInfo::writeDB:  "+e.getMessage()));
  }

  // Now get the ID
  if (!this->fetchID()) {
    throw(runtime_error("FEConfigPedInfo::writeDB:  Failed to write"));
  }
  
  return m_ID;
}




