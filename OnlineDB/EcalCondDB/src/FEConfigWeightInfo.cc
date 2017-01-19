#include <stdexcept>
#include <string>
#include <string.h>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include <cstdlib>
#include "OnlineDB/EcalCondDB/interface/FEConfigWeightInfo.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;

FEConfigWeightInfo::FEConfigWeightInfo()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;
  m_config_tag="";
  m_version=0;
  m_ID=0;
  clear();   
}


void FEConfigWeightInfo::clear(){
   m_ngr=0;
}



FEConfigWeightInfo::~FEConfigWeightInfo()
{
}



int FEConfigWeightInfo::fetchNextId()  noexcept(false) {

  int result=0;
  try {
    this->checkConnection();

    m_readStmt = m_conn->createStatement(); 
    m_readStmt->setSQL("select FE_CONFIG_WEIGHT_SQ.NextVal from DUAL ");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next ()){
      result= rset->getInt(1);
    }
    result++;
    m_conn->terminateStatement(m_readStmt);
    return result; 

  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigWeightInfo::fetchNextId():  "+e.getMessage()));
  }

}

void FEConfigWeightInfo::prepareWrite()
  noexcept(false)
{
  this->checkConnection();

  int next_id=0;
  if(getId()==0){
    next_id=fetchNextId();
  }

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO "+getTable()+" ( wei_conf_id, tag, version, number_of_groups) " 
			" VALUES ( :1, :2, :3 , :4) " );

    m_writeStmt->setInt(1, next_id);
    m_ID=next_id;

  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigWeightInfo::prepareWrite():  "+e.getMessage()));
  }

}

void FEConfigWeightInfo::setParameters(const std::map<string,string>& my_keys_map){
  
  // parses the result of the XML parser that is a map of 
  // string string with variable name variable value 
  
  for( std::map<std::string, std::string >::const_iterator ci=
	 my_keys_map.begin(); ci!=my_keys_map.end(); ci++ ) {
    
    if(ci->first==  "VERSION") setVersion(atoi(ci->second.c_str()) );
    if(ci->first==  "TAG") setConfigTag(ci->second);
    if(ci->first==  "NUMBER_OF_GROUPS") setNumberOfGroups(atoi(ci->second.c_str()) );
    
  }
  
}

void FEConfigWeightInfo::writeDB()
  noexcept(false)
{
  this->checkConnection();
  this->checkPrepare();

  try {

    // number 1 is the id 
    m_writeStmt->setString(2, this->getConfigTag());
    m_writeStmt->setInt(3, this->getVersion());
    m_writeStmt->setInt(4, this->getNumberOfGroups());

    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigWeightInfo::writeDB():  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("FEConfigWeightInfo::writeDB:  Failed to write"));
  }


}


void FEConfigWeightInfo::fetchData(FEConfigWeightInfo * result)
  noexcept(false)
{
  this->checkConnection();
  result->clear();
  if(result->getId()==0 && (result->getConfigTag()=="") ){
    throw(std::runtime_error("FEConfigWeightInfo::fetchData(): no Id defined for this FEConfigWeightInfo "));
  }

  try {
    DateHandler dh(m_env, m_conn);

    m_readStmt->setSQL("SELECT * FROM " + getTable() +   
                       " where ( wei_conf_id= :1 or (tag=:2 AND version=:3 ) )" );
    m_readStmt->setInt(1, result->getId());
    m_readStmt->setString(2, result->getConfigTag());
    m_readStmt->setInt(3, result->getVersion());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    // 1 is the id and 2 is the config tag and 3 is the version

    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));
    result->setVersion(rset->getInt(3));
    result->setNumberOfGroups(rset->getInt(4));
    Date dbdate = rset->getDate(5);
    result->setDBTime( dh.dateToTm( dbdate ));

  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigWeightInfo::fetchData():  "+e.getMessage()));
  }
}

void FEConfigWeightInfo::fetchLastData(FEConfigWeightInfo * result)
  noexcept(false)
{
  this->checkConnection();
  result->clear();
  try {
    DateHandler dh(m_env, m_conn);

    m_readStmt->setSQL("SELECT * FROM " + getTable() +   
                       " where   wei_conf_id = ( select max( wei_conf_id) from "+ getTable() +" ) " );
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));
    result->setVersion(rset->getInt(3));
    result->setNumberOfGroups(rset->getInt(4));
    Date dbdate = rset->getDate(5);
    result->setDBTime( dh.dateToTm( dbdate ));

  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigWeightInfo::fetchData():  "+e.getMessage()));
  }
}

int FEConfigWeightInfo::fetchID()    noexcept(false)
{
  // Return from memory if available
  if (m_ID!=0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT wei_conf_id FROM "+ getTable()+
                 " WHERE  tag=:1 and version=:2 " );

    stmt->setString(1, getConfigTag() );
    stmt->setInt(2, getVersion() );

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigWeightInfo::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}



void FEConfigWeightInfo::setByID(int id) 
  noexcept(false)
{
   this->checkConnection();

   DateHandler dh(m_env, m_conn);

   try {
     Statement* stmt = m_conn->createStatement();

     stmt->setSQL("SELECT * FROM fe_config_weight_info WHERE wei_conf_id = :1");
     stmt->setInt(1, id);
     
     ResultSet* rset = stmt->executeQuery();
     if (rset->next()) {
       this->setId(rset->getInt(1));
       this->setConfigTag(rset->getString(2));
       this->setVersion(rset->getInt(3));
       this->setNumberOfGroups(rset->getInt(4));
       Date dbdate = rset->getDate(5);
       this->setDBTime( dh.dateToTm( dbdate ));
     } else {
       throw(std::runtime_error("FEConfigWeightInfo::setByID:  Given config_id is not in the database"));
     }
     
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(std::runtime_error("FEConfigWeightInfo::setByID:  "+e.getMessage()));
   }
}



