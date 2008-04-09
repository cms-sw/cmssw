#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODLaserConfig.h"

using namespace std;
using namespace oracle::occi;

ODLaserConfig::ODLaserConfig()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;
  m_config_tag="";

   m_ID=0;
   clear();
}

void ODLaserConfig::clear(){


  m_debug=0;
  m_dummy=0;

  // emtc 
   m_emtc_1=0;
   m_emtc_2=0;
   m_emtc_3=0;
   m_emtc_4=0;
   m_emtc_5=0;

  // laser
  m_wave=0;
  m_power=0;
  m_switch=0;
  m_filter=0;
  m_on=0;
  m_laserhost="";
  m_laserport=0;

  // mataq
   m_mq_base=0;
   m_mq_none=0;
   m_mode="" ;
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
   m_aqmode="";
   m_mq_file="";
   m_laser_tag="";

}

ODLaserConfig::~ODLaserConfig()
{
}

int ODLaserConfig::fetchNextId()  throw(std::runtime_error) {

  int result=0;
  try {
    this->checkConnection();

    m_readStmt = m_conn->createStatement(); 
    m_readStmt->setSQL("select ecal_laser_config_sq.NextVal from dual");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next ()){
      result= rset->getInt(1);
    }
    m_conn->terminateStatement(m_readStmt);
    return result; 

  } catch (SQLException &e) {
    throw(runtime_error("ODLaserConfig::fetchNextId():  "+e.getMessage()));
  }

}


void ODLaserConfig::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();
  int next_id=fetchNextId();


  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_Laser_CONFIGURATION ( laser_configuration_id, laser_tag "
			", laser_DEBUG "
			", DUMMY "
			", MATACQ_BASE_ADDRESS " 
			", MATACQ_NONE "
			", matacq_mode "
			", channel_Mask "
			", max_Samples_For_Daq "
			", maTACQ_FED_ID "
			", pedestal_File "
			", use_Buffer "
			", postTrig "
			", fp_Mode "
			", hal_Module_File " 
			", hal_Address_Table_File "
			", hal_Static_Table_File "
			", matacq_Serial_Number "
			", pedestal_Run_Event_Count " 
			", raw_Data_Mode "
			", ACQUISITION_MODE " 
			", LOCAL_OUTPUT_FILE " 
			", emtc_none "
			", wte2_laser_delay " 
			", laser_phase "
			", emtc_ttc_in "
			", emtc_slot_id " 
			", WAVELENGTH "
			", POWER_SETTING "
			", OPTICAL_SWITCH "
			", FILTER "
			", LASER_CONTROL_ON " 
			", LASER_CONTROL_HOST " 
			", LASER_CONTROL_PORT "
			", LASER_TAG2 ) "
			"VALUES (  :1, :2, :3, :4, :5, :6, :7, :8, :9, :10, "
			":11, :12, :13, :14, :15, :16, :17, :18, :19, :20,  "
			":21, :22, :23, :24, :25, :26, :27, :28, :29, :30,  "
			":31, :32, :33, :34, :35 )");
    m_writeStmt->setInt(1, next_id);
    m_ID=next_id;
  } catch (SQLException &e) {
    throw(runtime_error("ODLaserConfig::prepareWrite():  "+e.getMessage()));
  }
}



void ODLaserConfig::writeDB()
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  try {
    
    // 1 is the id 2 is the tag
    m_writeStmt->setString(2, this->getConfigTag());

    m_writeStmt->setInt(   3, this->getDebug());
    m_writeStmt->setInt(   4, this->getDummy());
    m_writeStmt->setInt(   5, this->getMatacqBaseAddress());
    m_writeStmt->setInt(   6, this->getMatacqNone());
    m_writeStmt->setString(7, this->getMatacqMode());
    m_writeStmt->setInt(   8, this->getChannelMask());
    m_writeStmt->setString(9, this->getMaxSamplesForDaq());
    m_writeStmt->setInt(  10, this->getMatacqFedId());
    m_writeStmt->setString(11, this->getPedestalFile());
    m_writeStmt->setInt(  12, this->getUseBuffer());
    m_writeStmt->setInt(  13, this->getPostTrig());
    m_writeStmt->setInt(  14, this->getFPMode());
    m_writeStmt->setString(15,  this->getHalModuleFile() );
    m_writeStmt->setString(16, this->getHalAddressTableFile() );
    m_writeStmt->setString(17, this->getHalStaticTableFile() );
    m_writeStmt->setString(18, this->getMatacqSerialNumber() );
    m_writeStmt->setInt(   19, this->getPedestalRunEventCount() );
    m_writeStmt->setInt(   20, this->getRawDataMode());
    m_writeStmt->setString(21, this->getMatacqAcquisitionMode());
    m_writeStmt->setString(22, this->getLocalOutputFile());
    m_writeStmt->setInt(   23, this->getEMTCNone());
    m_writeStmt->setInt(   24, this->getWTE2LaserDelay());
    m_writeStmt->setInt(   25, this->getLaserPhase());
    m_writeStmt->setInt(   26, this->getEMTCTTCIn());
    m_writeStmt->setInt(   27, this->getEMTCSlotId());
    // laser
    m_writeStmt->setInt(28, this->getWaveLength());
    m_writeStmt->setInt(29, this->getPower());
    m_writeStmt->setInt(30, this->getOpticalSwitch());
    m_writeStmt->setInt(31, this->getFilter());
    m_writeStmt->setInt(32, this->getLaserControlOn());
    m_writeStmt->setString(33, this->getLaserControlHost() );
    m_writeStmt->setInt(   34, this->getLaserControlPort());
    m_writeStmt->setString(   35, this->getLaserTag());

    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(runtime_error("ODLaserConfig::writeDB():  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(runtime_error("ODLaserConfig::writeDB:  Failed to write"));
  }


}



void ODLaserConfig::fetchData(ODLaserConfig * result)
  throw(runtime_error)
{
  this->checkConnection();
  result->clear();
  if(result->getId()==0 && (result->getConfigTag()=="") ){
    throw(runtime_error("ODLaserConfig::fetchData(): no Id defined for this ODLaserConfig "));
  }

  try {

    m_readStmt->setSQL("SELECT * "
		       "FROM ECAL_Laser_CONFIGURATION d "
		       " where ( laser_configuration_id = :1  or laser_tag=:2 )" );
    m_readStmt->setInt(1, result->getId());
    m_readStmt->setString(2, result->getConfigTag());
   ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    
    // start from 3 because of select * 

    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));

    result->setDebug(rset->getInt(  3  ));
    result->setDummy(rset->getInt(  4  ));
    result->setMatacqBaseAddress(rset->getInt( 5   ));
    result->setMatacqNone(rset->getInt(  6  ));
    result->setMatacqMode(rset->getString(7    ));
    result->setChannelMask(rset->getInt(  8  ));
    result->setMaxSamplesForDaq(rset->getString( 9   ));
    result->setMatacqFedId(rset->getInt( 10   ));
    result->setPedestalFile(rset->getString( 11   ));
    result->setUseBuffer(rset->getInt(   12 ));
    result->setPostTrig(rset->getInt(   13 ));
    result->setFPMode(rset->getInt(   14 ));
    result->setHalModuleFile(rset->getString( 15   ));
    result->setHalAddressTableFile(rset->getString( 16   ));
    result->setHalStaticTableFile(rset->getString(  17  ));
    result->setMatacqSerialNumber(rset->getString(  18  ));
    result->setPedestalRunEventCount(rset->getInt(  19  ));
    result->setRawDataMode(rset->getInt( 20   ));
    result->setMatacqAcquisitionMode(rset->getString( 21   ));
    result->setLocalOutputFile(rset->getString(  22  ));
    result->setEMTCNone(rset->getInt(  23  ));
    result->setWTE2LaserDelay(rset->getInt( 24   ));
    result->setLaserPhase(rset->getInt(  25  ));
    result->setEMTCTTCIn(rset->getInt(  26  ));
    result->setEMTCSlotId(rset->getInt( 27   ));
    // laser
    result->setWaveLength(rset->getInt( 28   ));
    result->setPower(rset->getInt(  29  ));
    result->setOpticalSwitch(rset->getInt( 30   ));
    result->setFilter(rset->getInt(  31  ));
    result->setLaserControlOn(rset->getInt( 32   ));
    result->setLaserControlHost(rset->getString( 33   ));
    result->setLaserControlPort(rset->getInt( 34   ));
    result->setLaserTag(rset->getString( 35   ));
  
  } catch (SQLException &e) {
    throw(runtime_error("ODLaserConfig::fetchData():  "+e.getMessage()));
  }
}

int ODLaserConfig::fetchID()    throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID!=0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT laser_configuration_id FROM ecal_laser_configuration "
                 "WHERE laser_tag=:laser_tag ");
    stmt->setString(1, getLaserTag());

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("ODLaserConfig::fetchID:  "+e.getMessage()));
  }

  fetchData(this);

  return m_ID;
}
