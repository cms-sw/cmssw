#include <stdexcept>
#include <string>
#include <string.h>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include <cstdlib>
#include "OnlineDB/EcalCondDB/interface/ODGolBiasCurrentInfo.h"

using namespace std;
using namespace oracle::occi;

ODGolBiasCurrentInfo::ODGolBiasCurrentInfo()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;
  m_config_tag="";
   m_ID=0;
   m_version=0;
   clear();   
}


void ODGolBiasCurrentInfo::clear(){

}



ODGolBiasCurrentInfo::~ODGolBiasCurrentInfo()
{
}



int ODGolBiasCurrentInfo::fetchNextId()  throw(std::runtime_error) {

  int result=0;
  try {
    this->checkConnection();

    m_readStmt = m_conn->createStatement(); 
    m_readStmt->setSQL("select COND2CONF_INFO_SQ.NextVal from DUAL ");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next ()){
      result= rset->getInt(1);
    }
    result++;
    m_conn->terminateStatement(m_readStmt);
    return result; 

  } catch (SQLException &e) {
    throw(std::runtime_error("ODGolBiasCurrentInfo::fetchNextId():  "+e.getMessage()));
  }

}

void ODGolBiasCurrentInfo::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  int next_id=0;
  if(getId()==0){
    next_id=fetchNextId();
  } 

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO "+getTable()+" ( rec_id, tag, version) " 
			" VALUES ( :1, :2, :3 ) " );

    m_writeStmt->setInt(1, next_id);
    m_ID=next_id;

  } catch (SQLException &e) {
    throw(std::runtime_error("ODGolBiasCurrentInfo::prepareWrite():  "+e.getMessage()));
  }

}

void ODGolBiasCurrentInfo::setParameters(std::map<string,string> my_keys_map){
  
  // parses the result of the XML parser that is a map of 
  // string string with variable name variable value 
  
  for( std::map<std::string, std::string >::iterator ci=
	 my_keys_map.begin(); ci!=my_keys_map.end(); ci++ ) {
    
    if(ci->first==  "VERSION") setVersion(atoi(ci->second.c_str()) );
    if(ci->first==  "TAG") setConfigTag(ci->second);
    
  }
  
}

void ODGolBiasCurrentInfo::writeDB()
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  try {

    // number 1 is the id 
    m_writeStmt->setString(2, this->getConfigTag());
    m_writeStmt->setInt(3, this->getVersion());

    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(std::runtime_error("ODGolBiasCurrentInfo::writeDB():  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODGolBiasCurrentInfo::writeDB:  Failed to write"));
  } else {
    int old_version=this->getVersion();
    m_readStmt = m_conn->createStatement(); 
    this->fetchData (this);
    m_conn->terminateStatement(m_readStmt);
    if(this->getVersion()!=old_version) std::cout << "ODGolBiasCurrentInfo>>WARNING version is "<< getVersion()<< endl; 
  }


}


void ODGolBiasCurrentInfo::fetchData(ODGolBiasCurrentInfo * result)
  throw(std::runtime_error)
{
  this->checkConnection();
  result->clear();
  if(result->getId()==0 && (result->getConfigTag()=="") ){
    throw(std::runtime_error("ODGolBiasCurrentInfo::fetchData(): no Id defined for this ODGolBiasCurrentInfo "));
  }



  try {
    if(result->getId()!=0) { 
      m_readStmt->setSQL("SELECT * FROM " + getTable() +   
			 " where  rec_id = :1 ");
      m_readStmt->setInt(1, result->getId());
    } else if (result->getConfigTag()!="") {

      if(result->getVersion() !=0){
	m_readStmt->setSQL("SELECT * FROM " + getTable() +
		     " WHERE tag = :tag "
		     " and version = :version " );
	m_readStmt->setString(1, result->getConfigTag());
	m_readStmt->setInt(2, result->getVersion());
      } else {
	// always select the last inserted one with a given tag
	m_readStmt->setSQL("SELECT * FROM " + getTable() +
		     " WHERE tag = :1 and version= (select max(version) from "+getTable() +" where tag=:2) " );
	m_readStmt->setString(1, result->getConfigTag());
	m_readStmt->setString(2, result->getConfigTag());
      }

    } else {
      // we should never pass here 
      throw(std::runtime_error("ODGolBiasCurrentInfo::fetchData(): no Id defined for this record "));
    }


    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    // 1 is the id and 2 is the config tag and 3 is the version

    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));
    result->setVersion(rset->getInt(3));

  } catch (SQLException &e) {
    throw(std::runtime_error("ODGolBiasCurrentInfo::fetchData():  "+e.getMessage()));
  }
}

int ODGolBiasCurrentInfo::fetchID()    throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID!=0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT rec_id FROM "+ getTable()+
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
    throw(std::runtime_error("ODGolBiasCurrentInfo::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}
