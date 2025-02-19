#include <stdexcept>
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cstdlib>

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
  m_size=0;
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
   m_matacq_vernier_min=0;
   m_matacq_vernier_max=0;

   m_wte_2_led_delay =0;
   m_led1_on =0;
   m_led2_on =0;
   m_led3_on =0;
   m_led4_on =0;
   m_vinj =0;
   m_orange_led_mon_ampl=0;
   m_blue_led_mon_ampl =0;
   m_trig_log_file ="";
   m_led_control_on =0;
   m_led_control_host="";
   m_led_control_port =0;
   m_ir_laser_power =0;
   m_green_laser_power=0;
   m_red_laser_power =0;
   m_blue_laser_log_attenuator =0;
   m_ir_laser_log_attenuator =0;
   m_green_laser_log_attenuator  =0;
   m_red_laser_log_attenuator =0;
   m_laser_config_file ="";

}

ODLaserConfig::~ODLaserConfig()
{
}



void ODLaserConfig::setParameters(std::map<string,string> my_keys_map){
  
  // parses the result of the XML parser that is a map of 
  // string string with variable name variable value 
  
  for( std::map<std::string, std::string >::iterator ci=
	 my_keys_map.begin(); ci!=my_keys_map.end(); ci++ ) {
    
    if(ci->first==  "LASER_CONFIGURATION_ID") setConfigTag(ci->second);
    if(ci->first==  "DEBUG") setDebug(atoi(ci->second.c_str()) );
    if(ci->first==  "LASER_DEBUG") setDebug(atoi(ci->second.c_str()) );
    if(ci->first==  "DUMMY") setDummy(atoi(ci->second.c_str() ));
    if(ci->first==  "MATACQ_BASE_ADDRESS") setMatacqBaseAddress(atoi(ci->second.c_str() ));
    if(ci->first==  "MATACQ_NONE") setMatacqNone(atoi(ci->second.c_str() ));
    if(ci->first==  "MATACQ_MODE") setMatacqMode(ci->second);
    if(ci->first==  "CHANNEL_MASK") setChannelMask(atoi(ci->second.c_str() ));
    if(ci->first==  "MAX_SAMPLES_FOR_DAQ") setMaxSamplesForDaq(ci->second );
    if(ci->first==  "MATACQ_FED_ID") setMatacqFedId(atoi(ci->second.c_str()) );
    if(ci->first==  "PEDESTAL_FILE") setPedestalFile(ci->second );
    if(ci->first==  "USE_BUFFER") setUseBuffer(atoi(ci->second.c_str())) ;
    if(ci->first==  "POSTTRIG") setPostTrig(atoi(ci->second.c_str()) );
    if(ci->first==  "FP_MODE") setFPMode(atoi(ci->second.c_str() ));
    if(ci->first==  "HAL_MODULE_FILE") setHalModuleFile(ci->second );
    if(ci->first==  "HAL_ADDRESS_TABLE_FILE" || ci->first==  "HAL_ADDRESST_ABLE_FILE") setHalAddressTableFile(ci->second);
    if(ci->first==  "HAL_STATIC_TABLE_FILE") setHalStaticTableFile(ci->second );
    if(ci->first==  "MATACQ_SERIAL_NUMBER") setMatacqSerialNumber(ci->second );
    if(ci->first==  "PEDESTAL_RUN_EVENT_COUNT") setPedestalRunEventCount(atoi(ci->second.c_str()) );
    if(ci->first==  "RAW_DATA_MODE") setRawDataMode(atoi(ci->second.c_str()) );
    if(ci->first==  "ACQUISITION_MODE") setMatacqAcquisitionMode(ci->second );
    if(ci->first==  "LOCAL_OUTPUT_FILE") setLocalOutputFile(ci->second );
    if(ci->first==  "EMTC_NONE") setEMTCNone(atoi(ci->second.c_str()) );
    if(ci->first==  "WTE2_LASER_DELAY") setWTE2LaserDelay(atoi(ci->second.c_str()) );
    if(ci->first==  "LASER_PHASE") setLaserPhase(atoi(ci->second.c_str()) );
    if(ci->first==  "EMTC_TTC_IN") setEMTCTTCIn(atoi(ci->second.c_str()) );
    if(ci->first==  "EMTC_SLOT_ID") setEMTCSlotId(atoi(ci->second.c_str()) );
    if(ci->first==  "WAVELENGTH") setWaveLength(atoi(ci->second.c_str()) );
    if(ci->first==  "OPTICAL_SWITCH") setOpticalSwitch(atoi(ci->second.c_str()) );
    if(ci->first==  "POWER_SETTING") setPower(atoi(ci->second.c_str()) );
    if(ci->first==  "FILTER") setFilter(atoi(ci->second.c_str()) );
    if(ci->first==  "LASER_CONTROL_ON") setLaserControlOn(atoi(ci->second.c_str()) );
    if(ci->first==  "LASER_CONTROL_HOST") setLaserControlHost(ci->second );
    if(ci->first==  "LASER_CONTROL_PORT") setLaserControlPort(atoi(ci->second.c_str()) );
    if(ci->first==  "MATACQ_VERNIER_MAX") setMatacqVernierMax(atoi(ci->second.c_str()) );
    if(ci->first==  "MATACQ_VERNIER_MIN") setMatacqVernierMin(atoi(ci->second.c_str()) );

    if(ci->first==  "WTE_2_LED_DELAY") setWTE2LedDelay(atoi(ci->second.c_str()) );
    if(ci->first==  "LED1_ON") setLed1ON(atoi(ci->second.c_str()) );
    if(ci->first==  "LED2_ON") setLed2ON(atoi(ci->second.c_str()) );
    if(ci->first==  "LED3_ON") setLed3ON(atoi(ci->second.c_str()) );
    if(ci->first==  "LED4_ON") setLed4ON(atoi(ci->second.c_str()) );
    if(ci->first==  "VINJ") setVinj(atoi(ci->second.c_str()) );
    if(ci->first==  "ORANGE_LED_MON_AMPL") setOrangeLedMonAmpl(atoi(ci->second.c_str()) );
    if(ci->first==  "BLUE_LED_MON_AMPL") setBlueLedMonAmpl(atoi(ci->second.c_str()) );
    if(ci->first==  "TRIG_LOG_FILE") setTrigLogFile(ci->second.c_str() );
    if(ci->first==  "LED_CONTROL_ON") setLedControlON(atoi(ci->second.c_str()) );
    if(ci->first==  "LED_CONTROL_HOST") setLedControlHost( ci->second.c_str() );
    if(ci->first==  "LED_CONTROL_PORT") setLedControlPort(atoi(ci->second.c_str()) );
    if(ci->first==  "IR_LASER_POWER") setIRLaserPower(atoi(ci->second.c_str()) );
    if(ci->first==  "GREEN_LASER_POWER") setGreenLaserPower(atoi(ci->second.c_str()) );
    if(ci->first==  "RED_LASER_POWER") setRedLaserPower(atoi(ci->second.c_str()) );
    if(ci->first==  "BLUE_LASER_LOG_ATTENUATOR") setBlueLaserLogAttenuator(atoi(ci->second.c_str()) );
    if(ci->first==  "IR_LASER_LOG_ATTENUATOR") setIRLaserLogAttenuator(atoi(ci->second.c_str()) );
    if(ci->first==  "GREEN_LASER_LOG_ATTENUATOR") setGreenLaserLogAttenuator(atoi(ci->second.c_str()) );
    if(ci->first==  "RED_LASER_LOG_ATTENUATOR") setRedLaserLogAttenuator(atoi(ci->second.c_str()) );
  

    if(ci->first==  "LASER_CONFIG_FILE") {
      std::string fname=ci->second ;
      setLaserConfigFile(fname );
      // here we must open the file and read the DCC Clob
      std::cout << "Going to read Laser file: " << fname << endl;

      ifstream inpFile;
      inpFile.open(fname.c_str());

      // tell me size of file
      int bufsize = 0;
      inpFile.seekg( 0,ios::end );
      bufsize = inpFile.tellg();
      std::cout <<" bufsize ="<<bufsize<< std::endl;
      // set file pointer to start again
      inpFile.seekg( 0,ios::beg );

      m_size=bufsize;

      inpFile.close();

    }


  }
  
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
    throw(std::runtime_error("ODLaserConfig::fetchNextId():  "+e.getMessage()));
  }

}


void ODLaserConfig::prepareWrite()
  throw(std::runtime_error)
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
			", LASER_TAG2 "
			", MATACQ_VERNIER_MIN "
			", MATACQ_VERNIER_MAX "
			" , wte_2_led_delay " 
			" , led1_on "
			" , led2_on "
			" , led3_on "
			" , led4_on "
			" , VINJ "
			" , orange_led_mon_ampl" 
			" , blue_led_mon_ampl "
			" , trig_log_file "
			" , led_control_on "
			" , led_control_host "
			" , led_control_port "
			" , ir_laser_power "
			" , green_laser_power" 
			" , red_laser_power "
			" , blue_laser_log_attenuator "
			" , IR_LASER_LOG_ATTENUATOR "
			" , GREEN_LASER_LOG_ATTENUATOR"  
			" , RED_LASER_LOG_ATTENUATOR "
			" , LASER_CONFIG_FILE "
			" , laser_configuration ) "
			" VALUES (  :1, :2, :3, :4, :5, :6, :7, :8, :9, :10, "
			":11, :12, :13, :14, :15, :16, :17, :18, :19, :20,  "
			":21, :22, :23, :24, :25, :26, :27, :28, :29, :30,  "
			":31, :32, :33, :34, :35, :36, :37, "
			" :38, :39, :40, :41, :42, :43, :44, :45, :46, :47, :48, :49, :50, :51, :52, :53, :54, :55, :56, :57, :58  )");
    m_writeStmt->setInt(1, next_id);
    m_ID=next_id;
  } catch (SQLException &e) {
    throw(std::runtime_error("ODLaserConfig::prepareWrite():  "+e.getMessage()));
  }
}



void ODLaserConfig::writeDB()
  throw(std::runtime_error)
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

    m_writeStmt->setInt(   36, this->getMatacqVernierMin());
    m_writeStmt->setInt(   37, this->getMatacqVernierMax());


    // here goes the led and the new parameters 
    m_writeStmt->setInt(   38, this->getWTE2LedDelay());
    m_writeStmt->setInt(   39, this->getLed1ON());
    m_writeStmt->setInt(   40, this->getLed2ON());
    m_writeStmt->setInt(   41, this->getLed3ON());
    m_writeStmt->setInt(   42, this->getLed4ON());
    m_writeStmt->setInt(   43, this->getVinj());
    m_writeStmt->setInt(   44, this->getOrangeLedMonAmpl());
    m_writeStmt->setInt(   45, this->getBlueLedMonAmpl());
    m_writeStmt->setString(   46, this->getTrigLogFile());
    m_writeStmt->setInt(   47, this->getLedControlON());
    m_writeStmt->setString(   48, this->getLedControlHost());
    m_writeStmt->setInt(   49, this->getLedControlPort());
    m_writeStmt->setInt(   50, this->getIRLaserPower());
    m_writeStmt->setInt(   51, this->getGreenLaserPower());
    m_writeStmt->setInt(   52, this->getRedLaserPower());
    m_writeStmt->setInt(   53, this->getBlueLaserLogAttenuator());
    m_writeStmt->setInt(   54, this->getIRLaserLogAttenuator());
    m_writeStmt->setInt(   55, this->getGreenLaserLogAttenuator());
    m_writeStmt->setInt(   56, this->getRedLaserLogAttenuator());
    m_writeStmt->setString(   57, this->getLaserConfigFile());

    // and now the clob
    oracle::occi::Clob clob(m_conn);
    clob.setEmpty();
    m_writeStmt->setClob(58,clob);
    m_writeStmt->executeUpdate ();
    m_conn->terminateStatement(m_writeStmt);

    // now we read and update it
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL ("SELECT laser_configuration FROM "+getTable()+" WHERE"
                         " laser_configuration_id=:1 FOR UPDATE");
    std::cout<<"updating the laser clob "<<std::endl;
    

    m_writeStmt->setInt(1, m_ID);
    ResultSet* rset = m_writeStmt->executeQuery();
    rset->next ();
    oracle::occi::Clob clob_to_write = rset->getClob (1);
    cout << "Opening the clob in read write mode" << endl;

    populateClob (clob_to_write, getLaserConfigFile(), m_size);
    int clobLength=clob_to_write.length ();
    cout << "Length of the clob is: " << clobLength << endl;
    m_writeStmt->executeUpdate();
    m_writeStmt->closeResultSet (rset);


  } catch (SQLException &e) {
    throw(std::runtime_error("ODLaserConfig::writeDB():  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODLaserConfig::writeDB:  Failed to write"));
  }

}



void ODLaserConfig::fetchData(ODLaserConfig * result)
  throw(std::runtime_error)
{
  this->checkConnection();
  result->clear();
  if(result->getId()==0 && (result->getConfigTag()=="") ){
    throw(std::runtime_error("ODLaserConfig::fetchData(): no Id defined for this ODLaserConfig "));
  }

  try {

    m_readStmt->setSQL("SELECT * "
		       "FROM "+getTable()+
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
  
    result->setMatacqVernierMin(rset->getInt( 36   ));
    result->setMatacqVernierMax(rset->getInt( 37   ));

    result->setWTE2LedDelay(rset->getInt( 38   ));
    result->setLed1ON(rset->getInt( 39   ));
    result->setLed2ON(rset->getInt( 40   ));
    result->setLed3ON(rset->getInt( 41   ));
    result->setLed4ON(rset->getInt( 42   ));
    result->setVinj(rset->getInt( 43   ));
    result->setOrangeLedMonAmpl(rset->getInt( 44   ));
    result->setBlueLedMonAmpl(rset->getInt( 45   ));
    result->setTrigLogFile(rset->getString( 46   ));
    result->setLedControlON(rset->getInt( 47   ));
    result->setLedControlHost(rset->getString( 48   ));
    result->setLedControlPort(rset->getInt( 49   ));
    result->setIRLaserPower(rset->getInt( 50   ));
    result->setGreenLaserPower(rset->getInt( 51   ));
    result->setRedLaserPower(rset->getInt( 52   ));
    result->setBlueLaserLogAttenuator(rset->getInt( 53   ));
    result->setIRLaserLogAttenuator(rset->getInt( 54   ));
    result->setGreenLaserLogAttenuator(rset->getInt( 55   ));
    result->setRedLaserLogAttenuator(rset->getInt( 56   ));
    result->setLaserConfigFile(rset->getString( 57   ));

    Clob clob = rset->getClob (58);
    cout << "Opening the clob in Read only mode" << endl;
    clob.open (OCCI_LOB_READONLY);
    int clobLength=clob.length ();
    cout << "Length of the clob is: " << clobLength << endl;
    m_size=clobLength;
    unsigned char* buffer = readClob (clob, m_size);
    clob.close ();
    cout<< "the clob buffer is:"<<endl;
    for (int i = 0; i < clobLength; ++i)
      cout << (char) buffer[i];
    cout << endl;

    result->setLaserClob(buffer);


  } catch (SQLException &e) {
    throw(std::runtime_error("ODLaserConfig::fetchData():  "+e.getMessage()));
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
    throw(std::runtime_error("ODLaserConfig::fetchID:  "+e.getMessage()));
  }

  fetchData(this);

  return m_ID;
}
