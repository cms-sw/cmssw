#include <stdexcept>
#include <cstdlib>
#include <string>
#include <string.h>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODFEDAQConfig.h"

using namespace std;
using namespace oracle::occi;

#define MY_NULL -1
#define SET_INT( statement, paramNum, paramVal ) if( paramVal != MY_NULL ) { statement->setInt(paramNum, paramVal);  } else { statement->setNull(paramNum,OCCINUMBER); }
#define SET_STRING( statement, paramNum, paramVal ) if( ! paramVal.empty() ) { statement->setString(paramNum, paramVal); } else { statement->setNull(paramNum,OCCICHAR); }

int getInt(ResultSet * rset, int ipar )
{
  return  rset->isNull(ipar) ? MY_NULL : rset->getInt(ipar) ;
}

ODFEDAQConfig::ODFEDAQConfig()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;
  m_config_tag="";
   m_ID=0;
   clear();   
}


void ODFEDAQConfig::clear(){
   m_del=MY_NULL;
   m_wei=MY_NULL;
   m_ped=MY_NULL;

   m_bxt =MY_NULL;
   m_btt =MY_NULL;
   m_tbtt=MY_NULL;
   m_tbxt=MY_NULL;

   m_version=0;
   m_com="";
}



ODFEDAQConfig::~ODFEDAQConfig()
{
}



int ODFEDAQConfig::fetchNextId()  throw(std::runtime_error) {

  int result=0;
  try {
    this->checkConnection();

    m_readStmt = m_conn->createStatement(); 
    m_readStmt->setSQL("select fe_daq_conDfig_sq.nextVal from dual");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next ()){
      result= rset->getInt(1);
    }
    m_conn->terminateStatement(m_readStmt);
    return result; 

  } catch (SQLException &e) {
    throw(std::runtime_error("ODFEDAQConfig::fetchNextId():  "+e.getMessage()));
  }

}

void ODFEDAQConfig::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();
  int next_id=fetchNextId();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO FE_DAQ_CONFIG ( config_id, tag, version, ped_id, " 
			" del_id, wei_id,bxt_id, btt_id, tr_bxt_id, tr_btt_id, user_comment ) "
			"VALUES ( :1, :2, :3, :4, :5, :6, :7 ,:8, :9, :10, :11 )" );

    m_writeStmt->setInt(1, next_id);
    m_ID=next_id;

  } catch (SQLException &e) {
    throw(std::runtime_error("ODFEDAQConfig::prepareWrite():  "+e.getMessage()));
  }

}

void ODFEDAQConfig::setParameters(std::map<string,string> my_keys_map){
  
  // parses the result of the XML parser that is a map of 
  // string string with variable name variable value 
  
  for( std::map<std::string, std::string >::iterator ci=
	 my_keys_map.begin(); ci!=my_keys_map.end(); ci++ ) {
    
    if(ci->first==  "VERSION") setVersion(atoi(ci->second.c_str()) );
    if(ci->first==  "PED_ID") setPedestalId(atoi(ci->second.c_str()) );
    if(ci->first==  "DEL_ID") setDelayId(atoi(ci->second.c_str()));
    if(ci->first==  "WEI_ID") setWeightId(atoi(ci->second.c_str()));

    if(ci->first==  "BXT_ID") setBadXtId(atoi(ci->second.c_str()));
    if(ci->first==  "BTT_ID") setBadTTId(atoi(ci->second.c_str()));
    if(ci->first==  "TRIG_BXT_ID") setTriggerBadXtId(atoi(ci->second.c_str()));
    if(ci->first==  "TRIG_BTT_ID") setTriggerBadTTId(atoi(ci->second.c_str()));

    if(ci->first==  "COMMENT" || ci->first==  "USER_COMMENT") setComment(ci->second);
    
  }
  
}

void ODFEDAQConfig::writeDB()
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  try {

    // number 1 is the id 
    m_writeStmt->setString(2, this->getConfigTag());
    m_writeStmt->setInt(3, this->getVersion());
    SET_INT(m_writeStmt,4, this->getPedestalId());
    SET_INT(m_writeStmt,5, this->getDelayId());
    SET_INT(m_writeStmt,6, this->getWeightId());
    SET_INT(m_writeStmt,7, this->getBadXtId());
    SET_INT(m_writeStmt,8, this->getBadTTId());
    SET_INT(m_writeStmt,9, this->getTriggerBadXtId());
    SET_INT(m_writeStmt,10,this->getTriggerBadTTId());

    m_writeStmt->setString(11, this->getComment());

    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(std::runtime_error("ODFEDAQConfig::writeDB():  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODFEDAQConfig::writeDB:  Failed to write"));
  }


}


void ODFEDAQConfig::fetchData(ODFEDAQConfig * result)
  throw(std::runtime_error)
{
  this->checkConnection();
  result->clear();
  if(result->getId()==0 && (result->getConfigTag()=="") ){
    throw(std::runtime_error("ODFEDAQConfig::fetchData(): no Id defined for this ODFEDAQConfig "));
  }

  if(result->getConfigTag()!="" && result->getVersion() ==0  ){
    int new_version=0;
    std::cout<< "using new method : retrieving last version for this tag "<<endl;
    try {
      this->checkConnection();
      
      m_readStmt->setSQL("select max(version) from "+getTable()+" where tag=:tag " );
      m_readStmt->setString(1, result->getConfigTag());
      ResultSet* rset = m_readStmt->executeQuery();
      while (rset->next ()){
	new_version= rset->getInt(1);
      }
      m_conn->terminateStatement(m_readStmt);

      m_readStmt = m_conn->createStatement(); 
      
      result->setVersion(new_version);
      
    } catch (SQLException &e) {
      throw(std::runtime_error("ODFEDAQConfig::fetchData():  "+e.getMessage()));
    }
    
    
    
  }

  try {

    m_readStmt->setSQL("SELECT * FROM " + getTable() +   
                       " where ( config_id = :1 or (tag=:2 AND version=:3 ) )" );
    m_readStmt->setInt(1, result->getId());
    m_readStmt->setString(2, result->getConfigTag());
    m_readStmt->setInt(3, result->getVersion());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    // 1 is the id and 2 is the config tag and 3 is the version

    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));
    result->setVersion(rset->getInt(3));

    result->setPedestalId(       getInt(rset,4) );
    result->setDelayId(          getInt(rset,5) );
    result->setWeightId(         getInt(rset,6) );
    result->setBadXtId(          getInt(rset,7) );
    result->setBadTTId(          getInt(rset,8) );
    result->setTriggerBadXtId(   getInt(rset,9) );
    result->setTriggerBadTTId(   getInt(rset,10) );
    result->setComment(          rset->getString(11) );

  } catch (SQLException &e) {
    throw(std::runtime_error("ODFEDAQConfig::fetchData():  "+e.getMessage()));
  }
}

int ODFEDAQConfig::fetchID()    throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID!=0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT config_id FROM "+ getTable()+
                 "WHERE  tag=:1 and version=:2 " );

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
    throw(std::runtime_error("ODFEDAQConfig::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}
