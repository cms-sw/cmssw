#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODJBH4Config.h"

using namespace std;
using namespace oracle::occi;

ODJBH4Config::ODJBH4Config()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

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


void ODJBH4Config::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_Jbh4_CONFIGURATION ( "
			" useBuffer, halModuleFile, halAddressTableFile, halStaticTableFile, halcbd8210serialnumber, "
			" caenbridgetype, caenlinknumber, caenboardnumber) "
			" VALUES ( :1, :2, :3, :4, :5, :6, :7, :8 )");

  } catch (SQLException &e) {
    throw(runtime_error("ODJBH4Config::prepareWrite():  "+e.getMessage()));
  }
}



void ODJBH4Config::writeDB()
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  try {


    m_writeStmt->setInt(1, this->getUseBuffer());
    m_writeStmt->setString(2,  this->getHalModuleFile() );
    m_writeStmt->setString(3, this->getHalAddressTableFile() );
    m_writeStmt->setString(4, this->getHalStaticTableFile() );
    m_writeStmt->setString(5, this->getCbd8210SerialNumber() );
    m_writeStmt->setString(6, this->getCaenBridgeType() );
    m_writeStmt->setInt(7, this->getCaenLinkNumber() );
    m_writeStmt->setInt(8, this->getCaenBoardNumber() );
 
    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(runtime_error("ODJBH4Config::writeDB():  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(runtime_error("ODJBH4Config::writeDB:  Failed to write"));
  }

}




void ODJBH4Config::fetchData(ODJBH4Config * result)
  throw(runtime_error)
{
  this->checkConnection();
  result->clear();
  if(result->getId()==0){
    throw(runtime_error("ODJBH4Config::fetchData(): no Id defined for this ODJBH4Config "));
  }

  try {

    m_readStmt->setSQL("SELECT d.usebuffer, d.halmodulefile, d.haladdresstablefile, "
		       " d.halstatictablefile, d.halcbd8210serialnumber, d.caenbridgetype, d.caenlinknumber, d.caenboardnumber "
		       "FROM ECAL_Jbh4_CONFIGURATION d "
		       " where jbh4_configuration_id = :1 " );
    m_readStmt->setInt(1, result->getId());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();


    result->setUseBuffer(           rset->getInt(1) );
    result->setHalModuleFile(        rset->getString(2) );
    result->setHalAddressTableFile(         rset->getString(3) );
    result->setHalStaticTableFile(    rset->getString(4) );
    result->setCbd8210SerialNumber(        rset->getString(5) );
    result->setCaenBridgeType(           rset->getString(6) );
    result->setCaenLinkNumber(            rset->getInt(7) );
    result->setCaenBoardNumber(              rset->getInt(8) );

  } catch (SQLException &e) {
    throw(runtime_error("ODJBH4Config::fetchData():  "+e.getMessage()));
  }
}

int ODJBH4Config::fetchID()    throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID!=0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT jbh4_configuration_id FROM ecal_jbh4_configuration "
                 "WHERE  usebuffer=:1 AND  halmodulefile=:2 AND  haladdresstablefile=:3 AND  "
		 " halstatictablefile=:4 AND  halcbd8210serialnumber=:5 AND  caenbridgetype=:6 AND  caenlinknumber=:7 AND  caenboardnumber=:8");
    
    stmt->setInt(1, getUseBuffer());
    stmt->setString(2, getHalModuleFile());
    stmt->setString(3,getHalAddressTableFile());
    stmt->setString(4,getHalStaticTableFile());
    stmt->setString(5,getCbd8210SerialNumber());
    stmt->setString(6, getCaenBridgeType());
    stmt->setInt(7, getCaenLinkNumber());
    stmt->setInt(8, getCaenBoardNumber());

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("ODJBH4Config::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}
