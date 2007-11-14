#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigWeightsInfo.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;



FEConfigWeightsInfo::FEConfigWeightsInfo()
{
  m_conn = NULL;
  m_ID = 0;
  //
  m_ngr =0 ;
  m_tag ="";

}



FEConfigWeightsInfo::~FEConfigWeightsInfo(){}


void FEConfigWeightsInfo::setID(int id){ m_ID = id;  }
int FEConfigWeightsInfo::getID(){ return m_ID ;  }

void FEConfigWeightsInfo::setNumberOfGroups(int number_of_groups){ m_ngr = number_of_groups;  }
int FEConfigWeightsInfo::getNumberOfGroups()const {return m_ngr;  }


Tm FEConfigWeightsInfo::getDBTime() const{  return m_db_time;}

void FEConfigWeightsInfo::setTag(std::string x) {
  if (x != m_tag) {
    m_ID = 0;
    m_tag = x;
  }
}

std::string FEConfigWeightsInfo::getTag() const{  return m_tag;}


int FEConfigWeightsInfo::fetchID()
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
		 "WHERE number_of_groups = :number_of_groups  ");
    stmt->setInt(1, m_ngr);
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("FEConfigWeightsInfo::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}


void FEConfigWeightsInfo::setByID(int id) 
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
       throw(runtime_error("FEConfigWeightsInfo::setByID:  Given config_id is not in the database"));
     }
     
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(runtime_error("FEConfigWeightsInfo::setByID:  "+e.getMessage()));
   }
}



int FEConfigWeightsInfo::writeDB()
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
    
    stmt->setSQL("INSERT INTO FE_CONFIG_WEIGHT_INFO (wei_conf_id, number_of_groups, tag ) "
		 "VALUES (FE_CONFIG_WEIGHTS_SQ.NextVal, :1, :2 )");
    stmt->setInt(1, m_ngr);
    stmt->setString(2, m_tag );

    stmt->executeUpdate();

    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("FEConfigWeightsInfo::writeDB:  "+e.getMessage()));
  }

  // Now get the ID
  if (!this->fetchID()) {
    throw(runtime_error("FEConfigWeightsInfo::writeDB:  Failed to write"));
  }
  
  return m_ID;
}




