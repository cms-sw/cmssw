#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigSlidingInfo.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;



FEConfigSlidingInfo::FEConfigSlidingInfo()
{
  m_conn = NULL;
  m_ID = 0;
  //
  m_iov_id =0 ;
  m_tag ="";

}



FEConfigSlidingInfo::~FEConfigSlidingInfo(){}


void FEConfigSlidingInfo::setID(int id){ m_ID = id;  }
int FEConfigSlidingInfo::getID(){ return m_ID ;  }

void FEConfigSlidingInfo::setIOVId(int iov_id){ m_iov_id = iov_id;  }
int FEConfigSlidingInfo::getIOVId()const {return m_iov_id;  }


Tm FEConfigSlidingInfo::getDBTime() const{  return m_db_time;}

void FEConfigSlidingInfo::setTag(std::string x) {
  if (x != m_tag) {
    m_ID = 0;
    m_tag = x;
  }
}

std::string FEConfigSlidingInfo::getTag() const{  return m_tag;}


int FEConfigSlidingInfo::fetchID()
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
    stmt->setSQL("SELECT sli_conf_id FROM fe_config_sliding_info "
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
    throw(runtime_error("FEConfigSlidingInfo::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}


void FEConfigSlidingInfo::setByID(int id) 
  throw(std::runtime_error)
{
   this->checkConnection();

   DateHandler dh(m_env, m_conn);

   try {
     Statement* stmt = m_conn->createStatement();

     stmt->setSQL("SELECT iov_id, db_timestamp, tag FROM fe_config_sliding_info WHERE sli_conf_id = :1");
     stmt->setInt(1, id);
     
     ResultSet* rset = stmt->executeQuery();
     if (rset->next()) {
       m_iov_id = rset->getInt(1);
       Date dbdate = rset->getDate(2);
       m_db_time = dh.dateToTm( dbdate );
       m_tag=rset->getString(3);
       m_ID = id;

     } else {
       throw(runtime_error("FEConfigSlidingInfo::setByID:  Given config_id is not in the database"));
     }
     
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(runtime_error("FEConfigSlidingInfo::setByID:  "+e.getMessage()));
   }
}



int FEConfigSlidingInfo::writeDB()
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
    
    stmt->setSQL("INSERT INTO FE_CONFIG_SLIDING_INFO (sli_conf_id, iov_id, tag ) "
		 "VALUES (FE_CONFIG_SLIDING_SQ.NextVal, :1, :2 )");
    stmt->setInt(1, m_iov_id);
    stmt->setString(2, m_tag );

    stmt->executeUpdate();

    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("FEConfigSlidingInfo::writeDB:  "+e.getMessage()));
  }

  // Now get the ID
  if (!this->fetchID()) {
    throw(runtime_error("FEConfigSlidingInfo::writeDB:  Failed to write"));
  }
  
  return m_ID;
}




