#include <stdexcept>
#include <string>
#include <string.h>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include <cstdlib>
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;

FEConfigMainInfo::FEConfigMainInfo()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;
  m_config_tag="";
  m_ID=0;
  clear();   
}


void FEConfigMainInfo::clear(){
  m_ped=0;
  m_lin=0;
  m_lut=0;
  m_fgr=0;
  m_sli=0;
  m_wei=0;
  m_bxt=0;
  m_btt=0;
}



FEConfigMainInfo::~FEConfigMainInfo()
{
}



int FEConfigMainInfo::fetchNextId()  throw(std::runtime_error) {

  int result=0;
  try {
    this->checkConnection();

    m_readStmt = m_conn->createStatement(); 
    m_readStmt->setSQL("select FE_CONFIG_MAIN_SQ.NextVal from DUAL ");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next ()){
      result= rset->getInt(1);
    }
    result++;
    m_conn->terminateStatement(m_readStmt);
    return result; 

  } catch (SQLException &e) {
    throw(runtime_error("FEConfigMainInfo::fetchNextId():  "+e.getMessage()));
  }

}

void FEConfigMainInfo::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  int next_id=0;
  if(getId()==0){
    next_id=fetchNextId();
  }

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO "+getTable()+" (conf_id, ped_conf_id, lin_conf_id, lut_conf_id, fgr_conf_id, sli_conf_id, wei_conf_id, bxt_conf_id, btt_conf_id, tag ) " 
			" VALUES ( :1, :2, :3 , :4, :5, :6, :7, :8, :9, :10 ) " );

    m_writeStmt->setInt(1, next_id);
    m_ID=next_id;

  } catch (SQLException &e) {
    throw(runtime_error("FEConfigMainInfo::prepareWrite():  "+e.getMessage()));
  }

}

void FEConfigMainInfo::setParameters(std::map<string,string> my_keys_map){
  
  // parses the result of the XML parser that is a map of 
  // string string with variable name variable value 
  
  for( std::map<std::string, std::string >::iterator ci=
	 my_keys_map.begin(); ci!=my_keys_map.end(); ci++ ) {
    
    if(ci->first==  "TAG") setConfigTag(ci->second);
    if(ci->first==  "PED_CONF_ID") setPedId(atoi(ci->second.c_str()) );
    if(ci->first==  "LIN_CONF_ID") setLinId(atoi(ci->second.c_str()) );
    if(ci->first==  "LUT_CONF_ID") setLutId(atoi(ci->second.c_str()) );
    if(ci->first==  "FGR_CONF_ID") setFgrId(atoi(ci->second.c_str()) );
    if(ci->first==  "SLI_CONF_ID") setSliId(atoi(ci->second.c_str()) );
    if(ci->first==  "WEI_CONF_ID") setWeiId(atoi(ci->second.c_str()) );
    if(ci->first==  "BXT_CONF_ID") setBxtId(atoi(ci->second.c_str()) );
    if(ci->first==  "BTT_CONF_ID") setBttId(atoi(ci->second.c_str()) );
    
  }
  
}

void FEConfigMainInfo::writeDB()
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  try {

    // number 1 is the id 
    m_writeStmt->setInt(2, this->getPedId());
    m_writeStmt->setInt(3, this->getLinId());
    m_writeStmt->setInt(4, this->getLutId());
    m_writeStmt->setInt(5, this->getFgrId());
    m_writeStmt->setInt(6, this->getSliId());
    m_writeStmt->setInt(7, this->getWeiId());
    m_writeStmt->setInt(8, this->getBxtId());
    m_writeStmt->setInt(9, this->getBttId());
    m_writeStmt->setString(10, this->getConfigTag());


    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(runtime_error("FEConfigMainInfo::writeDB():  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(runtime_error("FEConfigMainInfo::writeDB:  Failed to write"));
  }


}


void FEConfigMainInfo::fetchData(FEConfigMainInfo * result)
  throw(runtime_error)
{
  this->checkConnection();
  result->clear();
  if(result->getId()==0 && (result->getConfigTag()=="") ){
    throw(runtime_error("FEConfigMainInfo::fetchData(): no Id defined for this FEConfigMainInfo "));
  }

  try {
    DateHandler dh(m_env, m_conn);

    m_readStmt->setSQL("SELECT * FROM " + getTable() +   
                       " where ( conf_id= :1 or tag=:2 )" );
    m_readStmt->setInt(1, result->getId());
    m_readStmt->setString(2, result->getConfigTag());

    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    // 1 is the id and 2 is the config tag and 3 is the version

    result->setId(rset->getInt(1));
    result->setPedId(rset->getInt(2));
    result->setLinId(rset->getInt(3));
    result->setLutId(rset->getInt(4));
    result->setFgrId(rset->getInt(5));
    result->setSliId(rset->getInt(6));
    result->setWeiId(rset->getInt(7));
    result->setBxtId(rset->getInt(8));
    result->setBttId(rset->getInt(9));
    result->setConfigTag(rset->getString(10));
    Date dbdate = rset->getDate(11);
    result->setDBTime( dh.dateToTm( dbdate ));

  } catch (SQLException &e) {
    throw(runtime_error("FEConfigMainInfo::fetchData():  "+e.getMessage()));
  }
}

void FEConfigMainInfo::fetchLastData(FEConfigMainInfo * result)
  throw(runtime_error)
{
  this->checkConnection();
  result->clear();
  try {
    DateHandler dh(m_env, m_conn);

    m_readStmt->setSQL("SELECT * FROM " + getTable() +   
                       " where   db_timestamp = ( select max( db_timestamp) from "+ getTable() +" ) " );
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setId(rset->getInt(1));
    result->setPedId(rset->getInt(2));
    result->setLinId(rset->getInt(3));
    result->setLutId(rset->getInt(4));
    result->setFgrId(rset->getInt(5));
    result->setSliId(rset->getInt(6));
    result->setWeiId(rset->getInt(7));
    result->setBxtId(rset->getInt(8));
    result->setBttId(rset->getInt(9));
    result->setConfigTag(rset->getString(10));
    Date dbdate = rset->getDate(11);
    result->setDBTime( dh.dateToTm( dbdate ));

  } catch (SQLException &e) {
    throw(runtime_error("FEConfigMainInfo::fetchData():  "+e.getMessage()));
  }
}

int FEConfigMainInfo::fetchID()    throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID!=0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT conf_id FROM "+ getTable()+
                 " WHERE  tag=:1 " );

    stmt->setString(1, getConfigTag() );


    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("FEConfigMainInfo::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}



void FEConfigMainInfo::setByID(int id) 
  throw(std::runtime_error)
{
   this->checkConnection();

   DateHandler dh(m_env, m_conn);

   try {
     Statement* stmt = m_conn->createStatement();

     stmt->setSQL("SELECT * FROM "+ getTable()+" WHERE conf_id = :1");
     stmt->setInt(1, id);
     
     ResultSet* rset = stmt->executeQuery();
     if (rset->next()) {
       this->setId(rset->getInt(1));
       this->setPedId(rset->getInt(2));
       this->setLinId(rset->getInt(3));
       this->setLutId(rset->getInt(4));
       this->setFgrId(rset->getInt(5));
       this->setSliId(rset->getInt(6));
       this->setWeiId(rset->getInt(7));
       this->setBxtId(rset->getInt(8));
       this->setBttId(rset->getInt(9));
       this->setConfigTag(rset->getString(10));
       Date dbdate = rset->getDate(11);
       this->setDBTime( dh.dateToTm( dbdate ));

     } else {
       throw(runtime_error("FEConfigMainInfo::setByID:  Given conf_id is not in the database"));
     }
     
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(runtime_error("FEConfigMainInfo::setByID:  "+e.getMessage()));
   }
}



