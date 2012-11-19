#include <stdexcept>
#include <string>
#include <sstream>
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
  m_db_checked = false;

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

  // new in 2012
  m_ir_laser_phase = 0;
  m_blue2_laser_phase = 0;
  m_blue_laser_post_trig = 0; 
  m_blue2_laser_post_trig = 0;
  m_ir_laser_post_trig = 0;
  m_green_laser_post_trig = 0;
  m_blue_laser_power = 0;
  m_wte2_blue_laser = 0; 
  m_wte2_blue2_laser = 0;
  m_wte2_ir_laser = 0;
  m_wte2_green_laser = 0;
  m_wte_2_led_soak_delay = 0;
  m_led_postscale = 0; 
  m_blue2_laser_power = 0;
  m_blue2_laser_log_attenuator = 0;
}

ODLaserConfig::~ODLaserConfig()
{
}

std::string ODLaserConfig::getLaserClobAsString() const {
  std::string ret;
  std::vector<unsigned char>::const_iterator i = m_laser_clob.begin();
  std::vector<unsigned char>::const_iterator e = m_laser_clob.end();
  while (i != e) {
    ret += *i++;
  }
  ret += "\0";
  return ret;
}

void ODLaserConfig::setLaserClob(unsigned char *s, int size) {
  m_laser_clob.clear();
  for (int i = 0; i < size; i++) {
    if (m_debug) {
      std::cout << "CLOB[" << i << "] = " << s[i] << " (" << 
	(unsigned int)s[i] << ")" << std::endl << std::flush;
    }
    m_laser_clob.push_back(s[i]);
  }
  if (m_debug) {
    std::cout << getLaserClobAsString() << std::endl << std::flush;
  }
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
    if(ci->first==  "BLUE_LASER_POSTTRIG") setBlueLaserPostTrig(atoi(ci->second.c_str()) );
    if(ci->first==  "BLUE2_LASER_POSTTRIG") setBlue2LaserPostTrig(atoi(ci->second.c_str()) );
    if(ci->first==  "IR_LASER_POSTTRIG") setIRLaserPostTrig(atoi(ci->second.c_str()) );
    if(ci->first==  "GREEN_LASER_POSTTRIG") setGreenLaserPostTrig(atoi(ci->second.c_str()) );
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
    if(ci->first==  "WTE_2_IR_LASER") setWTE2IRLaser(atoi(ci->second.c_str()) );
    if(ci->first==  "WTE_2_BLUE_LASER") setWTE2BlueLaser(atoi(ci->second.c_str()) );
    if(ci->first==  "WTE_2_BLUE2_LASER") setWTE2Blue2Laser(atoi(ci->second.c_str()) );
    if(ci->first==  "WTE_2_GREEN_LASER") setWTE2GreenLaser(atoi(ci->second.c_str()) );
    if(ci->first==  "LASER_PHASE") setLaserPhase(atoi(ci->second.c_str()) );
    if(ci->first==  "BLUE_LASER_PHASE") setBlueLaserPhase(atoi(ci->second.c_str()) );
    if(ci->first==  "BLUE2_LASER_PHASE") setBlue2LaserPhase(atoi(ci->second.c_str()) );
    if(ci->first==  "IR_LASER_PHASE") setIRLaserPhase(atoi(ci->second.c_str()) );
    if(ci->first==  "GREEN_LASER_PHASE") setGreenLaserPhase(atoi(ci->second.c_str()) );
    if(ci->first==  "EMTC_TTC_IN") setEMTCTTCIn(atoi(ci->second.c_str()) );
    if(ci->first==  "EMTC_SLOT_ID") setEMTCSlotId(atoi(ci->second.c_str()) );
    if(ci->first==  "WAVELENGTH") setWaveLength(atoi(ci->second.c_str()) );
    if(ci->first==  "OPTICAL_SWITCH") setOpticalSwitch(atoi(ci->second.c_str()) );
    if(ci->first==  "POWER_SETTING") setPower(atoi(ci->second.c_str()) );
    if(ci->first==  "BLUE_LASER_POWER") setBlueLaserPower(atoi(ci->second.c_str()) );
    if(ci->first==  "FILTER") setFilter(atoi(ci->second.c_str()) );
    if(ci->first==  "LASER_CONTROL_ON") setLaserControlOn(atoi(ci->second.c_str()) );
    if(ci->first==  "LASER_CONTROL_HOST") setLaserControlHost(ci->second );
    if(ci->first==  "LASER_CONTROL_PORT") setLaserControlPort(atoi(ci->second.c_str()) );
    if(ci->first==  "MATACQ_VERNIER_MAX") setMatacqVernierMax(atoi(ci->second.c_str()) );
    if(ci->first==  "MATACQ_VERNIER_MIN") setMatacqVernierMin(atoi(ci->second.c_str()) );
    
    if(ci->first==  "WTE_2_LED_DELAY") setWTE2LedDelay(atoi(ci->second.c_str()) );
    if(ci->first==  "WTE_2_LED_SOAK_DELAY") setWTE2LedSoakDelay(atoi(ci->second.c_str()) );
    if(ci->first==  "LED_POSTSCALE") setLedPostScale(atoi(ci->second.c_str()) );
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
    if(ci->first==  "BLUE2_LASER_POWER") setBlue2LaserPower(atoi(ci->second.c_str()) );
    if(ci->first==  "BLUE_LASER_LOG_ATTENUATOR") setBlueLaserLogAttenuator(atoi(ci->second.c_str()) );
    if(ci->first==  "BLUE2_LASER_LOG_ATTENUATOR") setBlue2LaserLogAttenuator(atoi(ci->second.c_str()) );
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
      if (m_debug) {
	std::cout <<" bufsize ="<<bufsize<< std::endl;
      }
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
    setDB();

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

std::string ODLaserConfig::values(int n) {
  std::stringstream r;
  for (int i = 0; i < n - 1; i++) {
    r << ":" << (i + 1)  << ", ";
  }
  r << ":" << n;
  return r.str();
}

void ODLaserConfig::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();
  setDB();
  int next_id=fetchNextId();


  try {
    m_writeStmt = m_conn->createStatement();
    std::string sql = "INSERT INTO " + getTable() + " ( "
      "  laser_configuration_id, laser_tag "
      ", laser_DEBUG "
      ", DUMMY "
      ", MATACQ_BASE_ADDRESS " 
      ", MATACQ_NONE "
      ", matacq_mode "
      ", channel_Mask "
      ", max_Samples_For_Daq "
      ", maTACQ_FED_ID "
      ", pedestal_File "
      ", use_Buffer ";
    if (m_isOldDb) {
      sql += ", postTrig ";
    } else {
      sql += ", blue_laser_posttrig ";
    }
    sql += ", fp_Mode "
      ", hal_Module_File " 
      ", hal_Address_Table_File "
      ", hal_Static_Table_File "
      ", matacq_Serial_Number "
      ", pedestal_Run_Event_Count " 
      ", raw_Data_Mode "
      ", ACQUISITION_MODE " 
      ", LOCAL_OUTPUT_FILE " 
      ", MATACQ_VERNIER_MIN "
      ", MATACQ_VERNIER_MAX "
      ", emtc_none ";
    if (m_isOldDb) {
      sql += ", wte2_laser_delay " ;
      sql += ", laser_phase ";
    }  else {
      sql += ", wte_2_blue_laser ";
      sql += ", blue_laser_phase ";
    }
    sql += ", emtc_ttc_in "
      ", emtc_slot_id " 
      ", WAVELENGTH ";
    if (m_isOldDb) {
      sql += ", POWER_SETTING";
    } else {
      sql += ", BLUE_LASER_POWER";
    }
    sql += ", OPTICAL_SWITCH "
      ", FILTER "
      ", LASER_CONTROL_ON " 
      ", LASER_CONTROL_HOST " 
      ", LASER_CONTROL_PORT "
      ", LASER_TAG2 "
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
      " , green_laser_power";
    if (m_isOldDb) {
      sql += " , red_laser_power ";
    } else {
      sql += ", blue2_laser_power";
    }
    sql += " , blue_laser_log_attenuator "
      " , IR_LASER_LOG_ATTENUATOR "
      " , GREEN_LASER_LOG_ATTENUATOR"  ;
    if (m_isOldDb) {
      sql += " , RED_LASER_LOG_ATTENUATOR ";
    } else {
      sql += " , BLUE2_LASER_LOG_ATTENUATOR"  ;
    }
    sql += " , LASER_CONFIG_FILE ";
    if (!m_isOldDb) {
      sql += ", IR_LASER_POSTTRIG"
	", BLUE2_LASER_POSTTRIG"
	", GREEN_LASER_POSTTRIG"
	", IR_LASER_PHASE"
	", BLUE2_LASER_PHASE"
	", GREEN_LASER_PHASE"
	", WTE_2_LED_SOAK_DELAY"
	", LED_POSTSCALE"
	", WTE_2_IR_LASER"
	", WTE_2_BLUE2_LASER"
	", WTE_2_GREEN_LASER"
	", laser_configuration) VALUES (";
      sql += values(69) + ")";
    } else {
      sql += ", laser_configuration) VALUES ("; 
      sql += values(58) + ")";
    }
    m_writeStmt->setSQL(sql);  
    m_writeStmt->setInt(1, next_id);
    m_ID=next_id;
    if (m_debug) {
      std::cout << "Executing query: " << std::endl << sql << std::endl;
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("ODLaserConfig::prepareWrite():  "+e.getMessage()));
  }
}

void ODLaserConfig::writeDB()
  throw(std::runtime_error)
{
  this->checkConnection();
  setDB();
  this->checkPrepare();

  try {
    
    // 1 is the id 2 is the tag
    int ifield = 2;
    m_writeStmt->setString(ifield++, this->getConfigTag());
    m_writeStmt->setInt(   ifield++, this->getDebug());
    m_writeStmt->setInt(   ifield++, this->getDummy());
    m_writeStmt->setInt(   ifield++, this->getMatacqBaseAddress());
    m_writeStmt->setInt(   ifield++, this->getMatacqNone());
    m_writeStmt->setString(ifield++, this->getMatacqMode());
    m_writeStmt->setInt(   ifield++, this->getChannelMask());
    m_writeStmt->setString(ifield++, this->getMaxSamplesForDaq());
    m_writeStmt->setInt(   ifield++, this->getMatacqFedId());
    m_writeStmt->setString(ifield++, this->getPedestalFile());
    m_writeStmt->setInt(   ifield++, this->getUseBuffer());
    if (!m_isOldDb) {
      m_writeStmt->setInt(   ifield++, this->getBlueLaserPostTrig());
    } else {
      m_writeStmt->setInt(   ifield++, this->getPostTrig());
    }
    m_writeStmt->setInt(   ifield++, this->getFPMode());
    m_writeStmt->setString(ifield++, this->getHalModuleFile() );
    m_writeStmt->setString(ifield++, this->getHalAddressTableFile() );
    m_writeStmt->setString(ifield++, this->getHalStaticTableFile() );
    m_writeStmt->setString(ifield++, this->getMatacqSerialNumber() );
    m_writeStmt->setInt(   ifield++, this->getPedestalRunEventCount() );
    m_writeStmt->setInt(   ifield++, this->getRawDataMode());
    m_writeStmt->setString(ifield++, this->getMatacqAcquisitionMode());
    m_writeStmt->setString(ifield++, this->getLocalOutputFile());
    m_writeStmt->setInt(   ifield++, this->getMatacqVernierMin());
    m_writeStmt->setInt(   ifield++, this->getMatacqVernierMax());
    m_writeStmt->setInt(   ifield++, this->getEMTCNone());
    if (!m_isOldDb) {
      m_writeStmt->setInt(   ifield++, this->getWTE2BlueLaser());
      m_writeStmt->setInt(   ifield++, this->getBlueLaserPhase());
    } else {
      m_writeStmt->setInt(   ifield++, this->getWTE2LaserDelay());
      m_writeStmt->setInt(   ifield++, this->getLaserPhase());
    }
    m_writeStmt->setInt(   ifield++, this->getEMTCTTCIn());
    m_writeStmt->setInt(   ifield++, this->getEMTCSlotId());
    // laser
    m_writeStmt->setInt(ifield++, this->getWaveLength());
    if (!m_isOldDb) {
      m_writeStmt->setInt(ifield++, this->getBlueLaserPower());
    } else {
      m_writeStmt->setInt(ifield++, this->getPower());
    }
    m_writeStmt->setInt(   ifield++, this->getOpticalSwitch());
    m_writeStmt->setInt(   ifield++, this->getFilter());
    m_writeStmt->setInt(   ifield++, this->getLaserControlOn());
    m_writeStmt->setString(ifield++, this->getLaserControlHost() );
    m_writeStmt->setInt(   ifield++, this->getLaserControlPort());
    m_writeStmt->setString(ifield++, this->getLaserTag());

    // here goes the led and the new parameters 
    m_writeStmt->setInt(   ifield++, this->getWTE2LedDelay());
    m_writeStmt->setInt(   ifield++, this->getLed1ON());
    m_writeStmt->setInt(   ifield++, this->getLed2ON());
    m_writeStmt->setInt(   ifield++, this->getLed3ON());
    m_writeStmt->setInt(   ifield++, this->getLed4ON());
    m_writeStmt->setInt(   ifield++, this->getVinj());
    m_writeStmt->setInt(   ifield++, this->getOrangeLedMonAmpl());
    m_writeStmt->setInt(   ifield++, this->getBlueLedMonAmpl());
    m_writeStmt->setString(ifield++, this->getTrigLogFile());
    m_writeStmt->setInt(   ifield++, this->getLedControlON());
    m_writeStmt->setString(ifield++, this->getLedControlHost());
    m_writeStmt->setInt(   ifield++, this->getLedControlPort());
    m_writeStmt->setInt(   ifield++, this->getIRLaserPower());
    m_writeStmt->setInt(   ifield++, this->getGreenLaserPower());
    if (!m_isOldDb) {
      m_writeStmt->setInt(   ifield++, this->getBlue2LaserPower());
    } else {
      m_writeStmt->setInt(   ifield++, this->getRedLaserPower());
    }
    m_writeStmt->setInt(   ifield++, this->getBlueLaserLogAttenuator());
    m_writeStmt->setInt(   ifield++, this->getIRLaserLogAttenuator());
    m_writeStmt->setInt(   ifield++, this->getGreenLaserLogAttenuator());
    if (!m_isOldDb) {
      m_writeStmt->setInt(   ifield++, this->getBlue2LaserLogAttenuator());
    } else {
      m_writeStmt->setInt(   ifield++, this->getRedLaserLogAttenuator());
    }
    m_writeStmt->setString(   ifield++, this->getLaserConfigFile());
    // new parameters added in 2012
    if (!m_isOldDb) {
      m_writeStmt->setInt(   ifield++, this->getIRLaserPostTrig());
      m_writeStmt->setInt(   ifield++, this->getBlue2LaserPostTrig());
      m_writeStmt->setInt(   ifield++, this->getGreenLaserPostTrig());
      m_writeStmt->setInt(   ifield++, this->getIRLaserPhase());
      m_writeStmt->setInt(   ifield++, this->getIRLaserPhase());
      m_writeStmt->setInt(   ifield++, this->getBlue2LaserPhase());
      m_writeStmt->setInt(   ifield++, this->getWTE2LedSoakDelay());
      m_writeStmt->setInt(   ifield++, this->getLedPostScale());
      m_writeStmt->setInt(   ifield++, this->getWTE2IRLaser());
      m_writeStmt->setInt(   ifield++, this->getWTE2Blue2Laser());
      m_writeStmt->setInt(   ifield++, this->getWTE2GreenLaser());
    }
    // and now the clob
    oracle::occi::Clob clob(m_conn);
    clob.setEmpty();
    m_writeStmt->setClob(ifield++, clob);
    m_writeStmt->executeUpdate();

    // now we read and update it
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL ("SELECT laser_configuration FROM " + getTable() +
			 " WHERE"
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

bool ODLaserConfig::setDB() {
  if (m_debug) {
    std::cout << "============ checking which DB we are interrogating ====="
	      << std::endl << std::flush;
  }
  if (!m_db_checked) {
    if (m_debug) {
      std::cout << "             Not yet checked..." << std::endl << std::flush;
    }
    m_isOldDb = false;
    this->checkConnection();  
    m_db_checked = true;
    try {
      m_readStmt = m_conn->createStatement();
      m_readStmt->setSQL("SELECT * FROM ECAL_LASER_CONFIGURATION_CP WHERE ROWNUM = 1");
      ResultSet* rset = m_readStmt->executeQuery();
      if (rset != NULL) {
	rset->next(); // just to avoid compilation warnings
      }
      if (m_debug) {
	std::cout << "New DB structure >= 2012" << std::endl;
      }
      m_conn->terminateStatement(m_readStmt);
    }
    catch (SQLException &e) {
      // old database - the new table structure is not in place
      m_isOldDb = true;
      if (m_debug) {
	std::cout << "Old DB structure <= 2012" << std::endl;
      }
    }
  }
  if (m_debug) {
    std::cout << "============ checked ============================= ====="
	      << std::endl << std::flush;
  }
  return m_isOldDb;
}

void ODLaserConfig::fetchData(ODLaserConfig * result)
  throw(std::runtime_error)
{
  setDB();
  this->checkConnection();
  createReadStatement();
  result->clear();
  if(result->getId()==0 && (result->getConfigTag()=="") ){
    throw(std::runtime_error("ODLaserConfig::fetchData(): no Id defined for this ODLaserConfig "));
  }
  if (m_debug) {
    std::cout << "Fetching data..." << std::endl << std::flush;
  }
  try {
    m_readStmt->setSQL("SELECT *"
		       " FROM " + getTable() + 
		       " where (laser_configuration_id=:1 or laser_tag=:2 )" );
    m_readStmt->setInt(1, result->getId());
    m_readStmt->setString(2, result->getConfigTag());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    int c = 1;
    result->setId(rset->getInt(c++));
    result->setConfigTag(rset->getString(c++));

    result->setDebug(rset->getInt(  c++  ));
    result->setDummy(rset->getInt(  c++  ));
    result->setMatacqBaseAddress(rset->getInt( c++   ));
    result->setMatacqNone(rset->getInt(  c++  ));
    result->setMatacqMode(rset->getString(c++    ));
    result->setChannelMask(rset->getInt(  c++  ));
    result->setMaxSamplesForDaq(rset->getString( c++   ));
    result->setMatacqFedId(rset->getInt( c++   ));
    result->setPedestalFile(rset->getString( c++   ));
    result->setUseBuffer(rset->getInt(   c++ ));
    if (!m_isOldDb) {
      result->setPostTrig(rset->getInt(   c++ ));
    } else {
      result->setBlueLaserPostTrig(rset->getInt(   c++ ));
    }
    result->setFPMode(rset->getInt(   c++ ));
    result->setHalModuleFile(rset->getString( c++   ));
    result->setHalAddressTableFile(rset->getString( c++   ));
    result->setHalStaticTableFile(rset->getString(  c++  ));
    result->setMatacqSerialNumber(rset->getString(  c++  ));
    result->setPedestalRunEventCount(rset->getInt(  c++  ));
    result->setRawDataMode(rset->getInt( c++   ));
    result->setMatacqAcquisitionMode(rset->getString( c++   ));
    result->setLocalOutputFile(rset->getString(  c++  ));
    result->setMatacqVernierMin(rset->getInt( c++   ));
    result->setMatacqVernierMax(rset->getInt( c++   ));
    result->setEMTCNone(rset->getInt(  c++  ));
    if (!m_isOldDb) {
      result->setWTE2LaserDelay(rset->getInt( c++   ));
      result->setLaserPhase(rset->getInt(  c++  ));
    } else {
      result->setWTE2BlueLaser(rset->getInt( c++   ));
      result->setBlueLaserPhase(rset->getInt(  c++  ));
    }
    result->setEMTCTTCIn(rset->getInt(  c++  ));
    result->setEMTCSlotId(rset->getInt( c++   ));
    // laser
    result->setWaveLength(rset->getInt( c++   ));
    if (!m_isOldDb) {
      result->setBlueLaserPower(rset->getInt(  c++  ));
    } else {
      result->setPower(rset->getInt(  c++  ));
    }
    result->setOpticalSwitch(rset->getInt( c++   ));
    result->setFilter(rset->getInt(  c++  ));
    result->setLaserControlOn(rset->getInt( c++   ));
    result->setLaserControlHost(rset->getString( c++   ));
    result->setLaserControlPort(rset->getInt( c++   ));
    result->setLaserTag(rset->getString( c++   ));
  
    result->setWTE2LedDelay(rset->getInt( c++   ));
    result->setLed1ON(rset->getInt( c++   ));
    result->setLed2ON(rset->getInt( c++   ));
    result->setLed3ON(rset->getInt( c++   ));
    result->setLed4ON(rset->getInt( c++   ));
    result->setVinj(rset->getInt( c++   ));
    result->setOrangeLedMonAmpl(rset->getInt( c++   ));
    result->setBlueLedMonAmpl(rset->getInt( c++   ));
    result->setTrigLogFile(rset->getString( c++   ));
    result->setLedControlON(rset->getInt( c++   ));
    result->setLedControlHost(rset->getString( c++   ));
    result->setLedControlPort(rset->getInt( c++   ));
    result->setIRLaserPower(rset->getInt( c++   ));
    result->setGreenLaserPower(rset->getInt( c++   ));
    if (!m_isOldDb) {
      result->setBlue2LaserPower(rset->getInt( c++   ));
    } else {
      result->setRedLaserPower(rset->getInt( c++   ));
    }
    result->setBlueLaserLogAttenuator(rset->getInt( c++   ));
    result->setIRLaserLogAttenuator(rset->getInt( c++   ));
    result->setGreenLaserLogAttenuator(rset->getInt( c++   ));
    if (!m_isOldDb) {
      result->setBlue2LaserLogAttenuator(rset->getInt( c++   ));
    } else {
      result->setRedLaserLogAttenuator(rset->getInt( c++   ));
    }
    result->setLaserConfigFile(rset->getString( c++   ));

    Clob clob = rset->getClob (c++);
    cout << "Opening the clob in Read only mode" << endl;
    clob.open (OCCI_LOB_READONLY);
    int clobLength=clob.length ();
    cout << "Length of the clob is: " << clobLength << endl;
    m_size=clobLength;
    unsigned char* buffer = readClob (clob, m_size);
    clob.close ();
    if (m_debug) {
      cout<< "the clob buffer is:"<<endl;
      for (int i = 0; i < clobLength; ++i) {
	cout << (char) buffer[i];
      }
      cout << endl;
    } 
    result->setLaserClob(buffer, clobLength);
    // parameters added in 2012
    if (!m_isOldDb) {
      result->setIRLaserPostTrig(rset->getInt( c++   ));
      result->setBlue2LaserPostTrig(rset->getInt( c++   ));
      result->setGreenLaserPostTrig(rset->getInt( c++   ));
      result->setIRLaserPhase(rset->getInt( c++   ));
      result->setBlue2LaserPhase(rset->getInt( c++   ));
      result->setGreenLaserPhase(rset->getInt( c++   ));
      result->setWTE2LedSoakDelay(rset->getInt( c++   ));
      result->setLedPostScale(rset->getInt( c++   ));
      result->setWTE2IRLaser(rset->getInt( c++   ));
      result->setWTE2Blue2Laser(rset->getInt( c++   ));
      result->setWTE2GreenLaser(rset->getInt( c++   ));
    }
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
  setDB();
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT laser_configuration_id FROM " + getTable() + 
                 " WHERE laser_tag=:laser_tag ");
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
