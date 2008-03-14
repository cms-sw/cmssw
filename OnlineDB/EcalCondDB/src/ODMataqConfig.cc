#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODMataqConfig.h"

using namespace std;
using namespace oracle::occi;

ODMataqConfig::ODMataqConfig()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_ID=0;
  clear();
}



ODMataqConfig::~ODMataqConfig()
{
}

void ODMataqConfig::clear(){
  m_mode="" ;
  m_fast_ped =0;
  m_chan_mask=0;
  m_samples="";
  m_ped_file="";
  m_use_buffer=0;
  m_post_trig=0;
  m_fp_mode=0;
  m_hal_mod_file="";
  m_hal_add_file="";
  m_hal_tab_file="";
  m_serial="";
  m_ped_count=0;
  m_raw_mode=0;
}


void ODMataqConfig::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_Matacq_CONFIGURATION ( "
			" matacq_mode, fastPedestal, channelMask, maxSamplesForDaq, pedestalFile, useBuffer, postTrig, fpMode, "
			" halModuleFile, halAddressTableFile, halStaticTableFile, matacqSerialNumber, pedestalRunEventCount, rawDataMode  ) "
			" VALUES ( :matacq_mode, :fastPedestal, :channelMask, :maxSamplesForDaq, :pedestalFile, :useBuffer, :postTrig, :fpMode, "
			" :halModuleFile, :halAddressTableFile, :halStaticTableFile, :matacqSerialNumber, :pedestalRunEventCount, :rawDataMode )");

  } catch (SQLException &e) {
    throw(runtime_error("ODMataqConfig::prepareWrite():  "+e.getMessage()));
  }
}



void ODMataqConfig::writeDB()
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  try {

    m_writeStmt->setString(1, this->getMataqMode());
    m_writeStmt->setInt(2, this->getFastPedestal());
    m_writeStmt->setInt(3, this->getChannelMask());
    m_writeStmt->setString(4, this->getMaxSamplesForDaq());
    m_writeStmt->setString(5, this->getPedestalFile());
    m_writeStmt->setInt(6, this->getUseBuffer());
    m_writeStmt->setInt(7, this->getPostTrig());
    m_writeStmt->setInt(8, this->getFPMode());
    m_writeStmt->setString(9,  this->getHalModuleFile() );
    m_writeStmt->setString(10, this->getHalAddressTableFile() );
    m_writeStmt->setString(11, this->getHalStaticTableFile() );
    m_writeStmt->setString(12, this->getMataqSerialNumber() );
    m_writeStmt->setInt(13, this->getPedestalRunEventCount() );
    m_writeStmt->setInt(14, this->getRawDataMode());
 
    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(runtime_error("ODMataqConfig::writeDB():  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(runtime_error("ODMataqConfig::writeDB:  Failed to write"));
  }

}




void ODMataqConfig::fetchData(ODMataqConfig * result)
  throw(runtime_error)
{
  this->checkConnection();
  result->clear();
  if(result->getId()==0){
    throw(runtime_error("ODMataqConfig::fetchData(): no Id defined for this ODMataqConfig "));
  }

  try {

    m_readStmt->setSQL("SELECT   d.matacq_mode, d.fastPedestal, d.channelMask, d.maxSamplesForDaq, d.pedestalFile, d.useBuffer, d.postTrig, d.fpMode, "
			" d.halModuleFile, d.halAddressTableFile, d.halStaticTableFile, d.matacqSerialNumber, d.pedestalRunEventCount, d.rawDataMode "
		       "FROM ECAL_Matacq_CONFIGURATION d "
		       " where matacq_configuration_id = :1 " );
    m_readStmt->setInt(1, result->getId());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setMataqMode(           rset->getString(1) );
    result->setFastPedestal(        rset->getInt(2) );
    result->setChannelMask(         rset->getInt(3) );
    result->setMaxSamplesForDaq(    rset->getString(4) );
    result->setPedestalFile(        rset->getString(5) );
    result->setUseBuffer(           rset->getInt(6) );
    result->setPostTrig(            rset->getInt(7) );
    result->setFPMode(              rset->getInt(8) );
    result->setHalModuleFile(       rset->getString(9) );
    result->setHalAddressTableFile( rset->getString(10) );
    result->setHalStaticTableFile(  rset->getString(11) );
    result->setMataqSerialNumber(   rset->getString(12) );
    result->setPedestalRunEventCount(rset->getInt(13) );
    result->setRawDataMode(         rset->getInt(14) );

  } catch (SQLException &e) {
    throw(runtime_error("ODMataqConfig::fetchData():  "+e.getMessage()));
  }
}

int ODMataqConfig::fetchID()    throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID!=0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT matacq_configuration_id FROM ecal_matacq_configuration "
                 "WHERE matacq_mode=:1 AND fastPedestal=:2 AND channelMask=:3 AND maxSamplesForDaq=:4 AND pedestalFile=:5 AND useBuffer=:6 AND postTrig=:7 AND fpMode=:8 AND "
		 " halModuleFile=:9 AND halAddressTableFile=:10 AND halStaticTableFile=:11 AND matacqSerialNumber=:12 AND pedestalRunEventCount=:13 AND rawDataMode=:14 ");
    

  stmt->setString(1, getMataqMode());
  stmt->setInt(2, getFastPedestal());
  stmt->setInt(3, getChannelMask());
  stmt->setString(4, getMaxSamplesForDaq());
  stmt->setString(5, getPedestalFile());
  stmt->setInt(6, getUseBuffer());
  stmt->setInt(7, getPostTrig());
  stmt->setInt(8, getFPMode());
  stmt->setString(9, getHalModuleFile());
  stmt->setString(10,getHalAddressTableFile());
  stmt->setString(11,getHalStaticTableFile());
  stmt->setString(12,getMataqSerialNumber());
  stmt->setInt(13,getPedestalRunEventCount());
  stmt->setInt(14,getRawDataMode());

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("ODMataqConfig::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}
