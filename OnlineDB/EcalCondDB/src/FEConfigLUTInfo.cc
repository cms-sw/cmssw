#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigLUTInfo.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;



FEConfigLUTInfo::FEConfigLUTInfo()
{
  m_conn = NULL;
  m_ID = 0;
  //
  m_iov_id =0 ;
  m_tag ="";

}



FEConfigLUTInfo::~FEConfigLUTInfo(){}


void FEConfigLUTInfo::setID(int id){ m_ID = id;  }
int FEConfigLUTInfo::getID(){ return m_ID ;  }

void FEConfigLUTInfo::setIOVId(int iov_id){ m_iov_id = iov_id;  }
int FEConfigLUTInfo::getIOVId()const {return m_iov_id;  }


Tm FEConfigLUTInfo::getDBTime() const{  return m_db_time;}

void FEConfigLUTInfo::setTag(std::string x) {
  if (x != m_tag) {
    m_tag = x;
  }
}

std::string FEConfigLUTInfo::getTag() const{  return m_tag;}


int FEConfigLUTInfo::fetchID()
  throw(runtime_error)
{
  // Return from memory if available
  if (m_ID>0) {
    return m_ID;
  }

  this->checkConnection();


  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT lut_conf_id FROM fe_config_lut_info "
		 "WHERE iov_id = :iov_id " 
		 " and DB_TIMESTAMP = :db_time) " );
    stmt->setInt(1, m_iov_id);
    stmt->setDate(2, dh.tmToDate(m_db_time));

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("FEConfigLUTInfo::fetchID:  "+e.getMessage()));
  }
  setByID(m_ID);
  return m_ID;
}



int FEConfigLUTInfo::fetchIDLast()
  throw(runtime_error)
{

  this->checkConnection();

  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT max( lut_conf_id) FROM fe_config_lut_info "	);
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("FEConfigLUTInfo::fetchIDLast:  "+e.getMessage()));
  }

  setByID(m_ID);
  return m_ID;
}

//
int FEConfigLUTInfo::fetchIDFromTag()
  throw(runtime_error)
{
  // selects tha most recent config id with a given tag 

  this->checkConnection();

  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT max(lut_conf_id) FROM fe_config_lut_info "
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
    throw(runtime_error("FEConfigLUTInfo::fetchIDFromTag:  "+e.getMessage()));
  }
  setByID(m_ID);
  return m_ID;
}






void FEConfigLUTInfo::setByID(int id) 
  throw(std::runtime_error)
{
   this->checkConnection();

   DateHandler dh(m_env, m_conn);

   cout<< "checking id "<<id<<endl;

   try {
     Statement* stmt = m_conn->createStatement();

     stmt->setSQL("SELECT iov_id, db_timestamp, tag FROM fe_config_lut_info WHERE lut_conf_id = :1");
     stmt->setInt(1, id);
     
     ResultSet* rset = stmt->executeQuery();
     if (rset->next()) {
       m_iov_id = rset->getInt(1);
       Date dbdate = rset->getDate(2);
       m_db_time = dh.dateToTm( dbdate );
       m_tag=rset->getString(3);
       m_ID = id;

     } else {
       throw(runtime_error("FEConfigLUTInfo::setByID:  Given config_id is not in the database"));
     }

     
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(runtime_error("FEConfigLUTInfo::setByID:  "+e.getMessage()));
   }
}



int FEConfigLUTInfo::writeDB()
  throw(runtime_error)
{
  this->checkConnection();

  // Validate the data, use infinity-till convention
  DateHandler dh(m_env, m_conn);

  try {

    // first reserve an id
    Statement* stmtq = m_conn->createStatement();
    stmtq->setSQL("SELECT FE_CONFIG_LUT_SQ.NextVal FROM dual "	);
    ResultSet* rsetq = stmtq->executeQuery();
    if (rsetq->next()) {
      m_ID = rsetq->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmtq);

    cout<< "going to use id "<<m_ID<<endl;

    // now insert 
    Statement* stmt = m_conn->createStatement();
    
    stmt->setSQL("INSERT INTO FE_CONFIG_LUT_INFO (lut_conf_id, iov_id, tag ) "
		 "VALUES (:1, :2, :3 )");
    stmt->setInt(1, m_ID);
    stmt->setInt(2, m_iov_id);
    stmt->setString(3, m_tag );

    stmt->executeUpdate();

    m_conn->terminateStatement(stmt);

    int ii=m_ID;
    setByID(ii);
    // this is to recover also the time info 


  } catch (SQLException &e) {
    throw(runtime_error("FEConfigLUTInfo::writeDB:  "+e.getMessage()));
  }

  cout<< "done inserting LUTInfo with id="<<m_ID<<endl;
  return m_ID;
}




