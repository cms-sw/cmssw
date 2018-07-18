#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODJBH4Config.h"

using namespace std;
using namespace oracle::occi;

ODJBH4Config::ODJBH4Config()
{
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;
  m_config_tag="";
  m_ID=0;
  clear();
}



ODJBH4Config::~ODJBH4Config()
{
}

void ODJBH4Config::clear(){

  m_use_buffer=0;
  m_hal_mod_file="";
  m_hal_add_file="";
  m_hal_tab_file="";
  m_serial="";
  m_caen1="";
  m_caen2=0;
  m_caen3=0;

}


int ODJBH4Config::fetchNextId()  noexcept(false) {

  int result=0;
  try {
    this->checkConnection();

    m_readStmt = m_conn->createStatement(); 
    m_readStmt->setSQL("select ecal_JBH4_config_sq.NextVal from dual");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next ()){
      result= rset->getInt(1);
    }
    m_conn->terminateStatement(m_readStmt);
    return result; 

  } catch (SQLException &e) {
    throw(std::runtime_error(std::string("ODJBH4Config::fetchNextId():  ")+getOraMessage(&e)));
  }

}

void ODJBH4Config::prepareWrite()
  noexcept(false)
{
  this->checkConnection();
  int next_id=fetchNextId();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_Jbh4_CONFIGURATION ( jbh4_configuration_id, jbh4_tag, "
			" useBuffer, halModuleFile, halAddressTableFile, halStaticTableFile, halcbd8210serialnumber, "
			" caenbridgetype, caenlinknumber, caenboardnumber) "
			" VALUES ( :1, :2, :3, :4, :5, :6, :7, :8 , :9, :10 )");

    m_writeStmt->setInt(1, next_id);
    m_ID=next_id;
  } catch (SQLException &e) {
    throw(std::runtime_error(std::string("ODJBH4Config::prepareWrite():  ")+getOraMessage(&e)));
  }
}



void ODJBH4Config::writeDB()
  noexcept(false)
{
  this->checkConnection();
  this->checkPrepare();

  try {

    // number 1 is the id number 2 is the tag
    m_writeStmt->setString(2, this->getConfigTag());

    m_writeStmt->setInt(3, this->getUseBuffer());
    m_writeStmt->setString(4,  this->getHalModuleFile() );
    m_writeStmt->setString(5, this->getHalAddressTableFile() );
    m_writeStmt->setString(6, this->getHalStaticTableFile() );
    m_writeStmt->setString(7, this->getCbd8210SerialNumber() );
    m_writeStmt->setString(8, this->getCaenBridgeType() );
    m_writeStmt->setInt(9, this->getCaenLinkNumber() );
    m_writeStmt->setInt(10, this->getCaenBoardNumber() );
 
    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(std::runtime_error(std::string("ODJBH4Config::writeDB():  ")+getOraMessage(&e)));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODJBH4Config::writeDB:  Failed to write"));
  }

}


void ODJBH4Config::fetchData(ODJBH4Config * result)
  noexcept(false)
{
  this->checkConnection();
  result->clear();
  if(result->getId()==0){
    throw(std::runtime_error("ODJBH4Config::fetchData(): no Id defined for this ODJBH4Config "));
  }

  try {

    m_readStmt->setSQL("SELECT * FROM ECAL_Jbh4_CONFIGURATION  "
		       " where ( jbh4_configuration_id = :1 or jbh4_tag=:2 )");
    m_readStmt->setInt(1, result->getId());
    m_readStmt->setString(2, result->getConfigTag());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setId(rset->getInt(1));
    result->setConfigTag(getOraString(rset,2));

    result->setUseBuffer(           rset->getInt(3) );
    result->setHalModuleFile(        getOraString(rset,4) );
    result->setHalAddressTableFile(         getOraString(rset,5) );
    result->setHalStaticTableFile(    getOraString(rset,6) );
    result->setCbd8210SerialNumber(        getOraString(rset,7) );
    result->setCaenBridgeType(           getOraString(rset,8) );
    result->setCaenLinkNumber(            rset->getInt(9) );
    result->setCaenBoardNumber(              rset->getInt(10) );

  } catch (SQLException &e) {
    throw(std::runtime_error(std::string("ODJBH4Config::fetchData():  ")+getOraMessage(&e)));
  }
}

int ODJBH4Config::fetchID()    noexcept(false)
{
  // Return from memory if available
  if (m_ID!=0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT jbh4_configuration_id FROM ecal_jbh4_configuration "
		 "WHERE  jbh4_tag=:jbh4_tag ");
    

    stmt->setString(1, getConfigTag());

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error(std::string("ODJBH4Config::fetchID:  ")+getOraMessage(&e)));
  }

  return m_ID;
}
