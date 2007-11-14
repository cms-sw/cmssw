#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigLinInfo.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;



FEConfigLinInfo::FEConfigLinInfo()
{
  m_conn = NULL;
  m_ID = 0;
  //
  m_iov_id =0 ;
  m_tag ="";

}



FEConfigLinInfo::~FEConfigLinInfo(){}


void FEConfigLinInfo::setID(int id){ m_ID = id;  }
int FEConfigLinInfo::getID(){ return m_ID ;  }

void FEConfigLinInfo::setIOVId(int iov_id){ m_iov_id = iov_id;  }
int FEConfigLinInfo::getIOVId()const {return m_iov_id;  }


Tm FEConfigLinInfo::getDBTime() const{  return m_db_time;}

void FEConfigLinInfo::setTag(std::string x) {
  if (x != m_tag) {
    m_ID = 0;
    m_tag = x;
  }
}

std::string FEConfigLinInfo::getTag() const{  return m_tag;}


int FEConfigLinInfo::fetchID()
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
    stmt->setSQL("SELECT lin_conf_id FROM fe_config_lin_info "
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
    throw(runtime_error("FEConfigLinInfo::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}


void FEConfigLinInfo::setByID(int id) 
  throw(std::runtime_error)
{
   this->checkConnection();

   DateHandler dh(m_env, m_conn);

   try {
     Statement* stmt = m_conn->createStatement();

     stmt->setSQL("SELECT iov_id, db_timestamp, tag FROM fe_config_lin_info WHERE lin_conf_id = :1");
     stmt->setInt(1, id);
     
     ResultSet* rset = stmt->executeQuery();
     if (rset->next()) {
       m_iov_id = rset->getInt(1);
       Date dbdate = rset->getDate(2);
       m_db_time = dh.dateToTm( dbdate );
       m_tag=rset->getString(3);
       m_ID = id;

     } else {
       throw(runtime_error("FEConfigLinInfo::setByID:  Given config_id is not in the database"));
     }
     
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(runtime_error("FEConfigLinInfo::setByID:  "+e.getMessage()));
   }
}



int FEConfigLinInfo::writeDB()
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
    
    stmt->setSQL("INSERT INTO FE_CONFIG_LIN_INFO (lin_conf_id, iov_id, tag ) "
		 "VALUES (FE_CONFIG_LIN_SQ.NextVal, :1, :2 )");
    stmt->setInt(1, m_iov_id);
    stmt->setString(2, m_tag );

    stmt->executeUpdate();

    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("FEConfigLinInfo::writeDB:  "+e.getMessage()));
  }

  // Now get the ID
  if (!this->fetchID()) {
    throw(runtime_error("FEConfigLinInfo::writeDB:  Failed to write"));
  }
  
  return m_ID;
}




