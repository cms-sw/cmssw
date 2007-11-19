#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigWeightInfo.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;



FEConfigWeightInfo::FEConfigWeightInfo()
{
  m_conn = NULL;
  m_ID = 0;
  //
  m_ngr =0 ;
  m_tag ="";

}



FEConfigWeightInfo::~FEConfigWeightInfo(){}


void FEConfigWeightInfo::setID(int id){ m_ID = id;  }
int FEConfigWeightInfo::getID(){ return m_ID ;  }

void FEConfigWeightInfo::setNumberOfGroups(int number_of_groups){ m_ngr = number_of_groups;  }
int FEConfigWeightInfo::getNumberOfGroups()const {return m_ngr;  }


Tm FEConfigWeightInfo::getDBTime() const{  return m_db_time;}

void FEConfigWeightInfo::setTag(std::string x) {
  if (x != m_tag) {
    m_tag = x;
  }
}

std::string FEConfigWeightInfo::getTag() const{  return m_tag;}


int FEConfigWeightInfo::fetchID()
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
    stmt->setSQL("SELECT wei_conf_id FROM fe_config_weight_info "
		 "WHERE DB_TIMESTAMP = :db_time) " );
    stmt->setDate(1, dh.tmToDate(m_db_time));
		 
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("FEConfigWeightInfo::fetchID:  "+e.getMessage()));
  }
  setByID(m_ID);
  return m_ID;
}



//
int FEConfigWeightInfo::fetchIDLast()
  throw(runtime_error)
{

  this->checkConnection();


  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT max( wei_conf_id) FROM fe_config_weight_info "    );
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("FEConfigWeightInfo::fetchIDLast:  "+e.getMessage()));
  }
  setByID(m_ID);

  return m_ID;
}

//
int FEConfigWeightInfo::fetchIDFromTag()
  throw(runtime_error)
{
  // selects tha most recent config id with a given tag

  this->checkConnection();

  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT max(wei_conf_id) FROM fe_config_weight_info "
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
    throw(runtime_error("FEConfigWeightInfo::fetchIDFromTag:  "+e.getMessage()));
  }
  setByID(m_ID);

  return m_ID;
}








void FEConfigWeightInfo::setByID(int id) 
  throw(std::runtime_error)
{
   this->checkConnection();

   DateHandler dh(m_env, m_conn);

   try {
     Statement* stmt = m_conn->createStatement();

     stmt->setSQL("SELECT number_of_groups, db_timestamp, tag FROM fe_config_weight_info WHERE wei_conf_id = :1");
     stmt->setInt(1, id);
     
     ResultSet* rset = stmt->executeQuery();
     if (rset->next()) {
       m_ngr = rset->getInt(1);
       Date dbdate = rset->getDate(2);
       m_db_time = dh.dateToTm( dbdate );
       m_tag=rset->getString(3);
       m_ID = id;

     } else {
       throw(runtime_error("FEConfigWeightInfo::setByID:  Given config_id is not in the database"));
     }
     
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(runtime_error("FEConfigWeightInfo::setByID:  "+e.getMessage()));
   }
}



int FEConfigWeightInfo::writeDB()
  throw(runtime_error)
{
  this->checkConnection();

  // Validate the data, use infinity-till convention
  DateHandler dh(m_env, m_conn);

  try {

    // first reserve an id
    Statement* stmtq = m_conn->createStatement();
    stmtq->setSQL("SELECT FE_CONFIG_WEIGHT_SQ.NextVal FROM dual "  );
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
    
    stmt->setSQL("INSERT INTO FE_CONFIG_WEIGHT_INFO ( wei_conf_id, number_of_groups, tag ) "
		 "VALUES (:1, :2 , :3 )");
    stmt->setInt(1, m_ID);
    stmt->setInt(2, m_ngr);
    stmt->setString(3, m_tag );

    stmt->executeUpdate();


    m_conn->terminateStatement(stmt);


    int ii=m_ID;
    setByID(ii);
    // this is to recover also the time info

  } catch (SQLException &e) {
    throw(runtime_error("FEConfigWeightInfo::writeDB:  "+e.getMessage()));
  }

  cout<< "done inserting LUTInfo with id="<<m_ID<<endl;
  
  return m_ID;
}




