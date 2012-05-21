#ifndef ODLASERCONFIG_H
#define ODLASERCONFIG_H

#include <map>
#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#define USE_NORM 1
#define USE_CHUN 2
#define USE_BUFF 3

/* Buffer Size */
#define BUFSIZE 200;

class ODLaserConfig : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  ODLaserConfig();
  ~ODLaserConfig();

  // User data methods
  inline std::string getTable() { 
    return "ECAL_LASER_CONFIGURATION";
  }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }
  inline void setSize(unsigned int id) { m_size = id; }
  inline unsigned int getSize() const { return m_size; }

  inline void setDebug() { setDebug(1); }
  inline void setDebug(int x) { m_debug = x; IODConfig::setDebug(); }
  inline void noDebug() { m_debug = 0; }
  inline int getDebug() const { return m_debug; }
  inline void setDummy(int x) { m_dummy = x; }
  inline int getDummy() const { return m_dummy; }

  inline void setMatacqBaseAddress(int x) { m_mq_base = x; }
  inline int getMatacqBaseAddress() const { return m_mq_base; }
  inline void setMatacqNone(int x) { m_mq_none = x; }
  inline int getMatacqNone() const { return m_mq_none; }
  inline void setMatacqMode(std::string x) { m_mode = x; }
  inline std::string getMatacqMode() const { return m_mode; }
  inline void setChannelMask(int x) { m_chan_mask = x; }
  inline int getChannelMask() const { return m_chan_mask; }
  inline void setMaxSamplesForDaq(std::string x) { m_samples = x; }
  inline std::string getMaxSamplesForDaq() const { return m_samples; }
  inline void setMatacqFedId(int x) { m_mq_fed = x; }
  inline int getMatacqFedId() const { return m_mq_fed; }
  inline void setPedestalFile(std::string x) { m_ped_file = x; }
  inline std::string getPedestalFile() const { return m_ped_file; }
  inline void setUseBuffer(int x) { m_use_buffer = x; }
  inline int getUseBuffer() const { return m_use_buffer; }
  inline void setPostTrig(int x) { 
    m_post_trig = x; 
    if (x != getBlueLaserPostTrig()) {
      setBlueLaserPostTrig(x);
    }
  }
  inline int getPostTrig() const { return m_post_trig; }
  // new in 2012
  inline void setBlueLaserPostTrig(int x) { 
    m_blue_laser_post_trig = x; 
    if (x != getPostTrig()) {
      setPostTrig(x);
    }
  }
  inline int getBlueLaserPostTrig() const { return m_blue_laser_post_trig; }
  inline void setBlue2LaserPostTrig(int x) { m_blue2_laser_post_trig = x; }
  inline int getBlue2LaserPostTrig() const { return m_blue2_laser_post_trig; }
  inline void setIRLaserPostTrig(int x) { m_ir_laser_post_trig = x; }
  inline int getIRLaserPostTrig() const { return m_ir_laser_post_trig; }
  inline void setGreenLaserPostTrig(int x) { m_green_laser_post_trig = x; }
  inline int getGreenLaserPostTrig() const { return m_green_laser_post_trig; }
  //
  inline void setFPMode(int x) { m_fp_mode = x; }
  inline int getFPMode() const { return m_fp_mode; }
  inline void setHalModuleFile(std::string x) { m_hal_mod_file = x; }
  inline std::string getHalModuleFile() const { return m_hal_mod_file; }
  inline void setMatacqVernierMax(int x) { m_matacq_vernier_max = x; } 
  inline int getMatacqVernierMax() { return m_matacq_vernier_max; } 
  inline void setMatacqVernierMin(int x) { m_matacq_vernier_min = x; } 
  inline int getMatacqVernierMin() { return m_matacq_vernier_min; } 
  
  inline void setHalAddressTableFile(std::string x) { m_hal_add_file = x; }
  inline std::string getHalAddressTableFile() const { return m_hal_add_file; }

  inline void setHalStaticTableFile(std::string x) { m_hal_tab_file = x; }
  inline std::string getHalStaticTableFile() const { return m_hal_tab_file; }

  inline void setMatacqSerialNumber(std::string x) { m_serial = x; }
  inline std::string getMatacqSerialNumber() const { return m_serial; }

  inline void setPedestalRunEventCount(int x) { m_ped_count = x; }
  inline int getPedestalRunEventCount() const { return m_ped_count; }

  inline void setRawDataMode(int x) { m_raw_mode = x; }
  inline int getRawDataMode() const { return m_raw_mode; }

  inline void setMatacqAcquisitionMode(std::string x) { m_aqmode = x; }
  inline std::string getMatacqAcquisitionMode() const { return m_aqmode; }

  inline void setLocalOutputFile(std::string x) { m_mq_file = x; }
  inline std::string getLocalOutputFile() const { return m_mq_file; }

  // emtc  
  inline void setEMTCNone(int x) { m_emtc_1 = x; }
  inline int getEMTCNone() const { return m_emtc_1; }
  inline void setWTE2LaserDelay(int x) { 
    m_emtc_2 = x; 
    if (x != getWTE2BlueLaser()) {
      setWTE2BlueLaser(x);
    }
  }
  inline int getWTE2LaserDelay() const { return m_emtc_2; }
  // new in 2012
  inline void setWTE2IRLaser(int x) { m_wte2_ir_laser = x; }
  inline int getWTE2IRLaser() const { return m_wte2_ir_laser; }
  inline void setWTE2BlueLaser(int x) { 
    m_wte2_blue_laser = x; 
    if (x != getWTE2LaserDelay()) {
      setWTE2LaserDelay(x);
    }
  }
  inline int getWTE2BlueLaser() const { return m_wte2_blue_laser; }
  inline void setWTE2Blue2Laser(int x) { m_wte2_blue2_laser = x; }
  inline int getWTE2Blue2Laser() const { return m_wte2_blue2_laser; }
  inline void setWTE2GreenLaser(int x) { m_wte2_green_laser = x; }
  inline int getWTE2GreenLaser() const { return m_wte2_green_laser; }
  //
  inline void setLaserPhase(int x) { 
    m_emtc_3 = x; 
    if (x != getBlueLaserPhase()) {
      setBlueLaserPhase(x);
    }
  }
  inline int getLaserPhase() const { return m_emtc_3; }
  // new in 2012
  inline void setBlueLaserPhase(int x) { 
    m_blue_laser_phase = x; 
    if (x != getLaserPhase()) {
      setLaserPhase(x);
    }
  }
  inline int getBlueLaserPhase() const { return m_blue_laser_phase; }
  inline void setBlue2LaserPhase(int x) { m_blue2_laser_phase = x; }
  inline int getBlue2LaserPhase() const { return m_blue2_laser_phase; }
  inline void setIRLaserPhase(int x) { m_ir_laser_phase = x; }
  inline int getIRLaserPhase() const { return m_ir_laser_phase; }
  inline void setGreenLaserPhase(int x) { m_green_laser_phase = x; }
  inline int getLGreenLaserPhase() const { return m_green_laser_phase; }
  //
  inline void setEMTCTTCIn(int x) { m_emtc_4 = x; }
  inline int getEMTCTTCIn() const { return m_emtc_4; }
  inline void setEMTCSlotId(int x) { m_emtc_5 = x; }
  inline int getEMTCSlotId() const { return m_emtc_5; }

  //  void setParameters(std::map<std::string,std::string> my_keys_map);

  // laser 

  inline void setWaveLength(int x) { m_wave = x; }
  inline int getWaveLength() const { return m_wave; }

  inline void setPower(int x) { m_power = x; }
  inline int getPower() const { return m_power; }
  // new in 2012
  inline void setBlueLaserPower(int x) { m_blue_laser_power = x; }
  inline int getBlueLaserPower() const { return m_blue_laser_power; }
  //
  inline void setOpticalSwitch(int x) { m_switch = x; }
  inline int getOpticalSwitch() const { return m_switch; }

  inline void setFilter(int x) { m_filter = x; }
  inline int getFilter() const { return m_filter; }

  inline void setLaserControlOn(int x) { m_on = x; }
  inline int getLaserControlOn() const { return m_on; }

  inline void setLaserControlHost(std::string x) { m_laserhost = x; }
  inline std::string getLaserControlHost() const { return m_laserhost; }

  inline void setLaserControlPort(int x) { m_laserport = x; }
  inline int getLaserControlPort() const { return m_laserport; }



  inline void setLaserTag(std::string x) { m_laser_tag = x; }
  inline std::string getLaserTag() const { return m_laser_tag ; }



  // new parameters 

  inline void setWTE2LedDelay(int x) { m_wte_2_led_delay = x; }
  inline int getWTE2LedDelay() const { return m_wte_2_led_delay; }

  // new in 2012
  inline void setWTE2LedSoakDelay(int x) { m_wte_2_led_soak_delay = x; }
  inline int getWTE2LedSoakDelay() { return m_wte_2_led_soak_delay; }
  inline void setLedPostScale(int x) { m_led_postscale = x; };
  inline int getLedPostScale() { return m_led_postscale; };
  //
  inline void setLed1ON(int x) { m_led1_on = x; }
  inline int getLed1ON() const { return m_led1_on; }

  inline void setLed2ON(int x) { m_led2_on = x; }
  inline int getLed2ON() const { return m_led2_on; }

  inline void setLed3ON(int x) { m_led3_on = x; }
  inline int getLed3ON() const { return m_led3_on; }

  inline void setLed4ON(int x) { m_led4_on = x; }
  inline int getLed4ON() const { return m_led4_on; }

  inline void setVinj(int x) { m_vinj = x; }
  inline int getVinj() const { return m_vinj; }

  inline void setOrangeLedMonAmpl(int x) { m_orange_led_mon_ampl = x; }
  inline int getOrangeLedMonAmpl() const { return m_orange_led_mon_ampl; }

  inline void setBlueLedMonAmpl(int x) { m_blue_led_mon_ampl = x; }
  inline int getBlueLedMonAmpl() const { return m_blue_led_mon_ampl; }

  inline void setTrigLogFile(std::string x) { m_trig_log_file = x; }
  inline std::string getTrigLogFile() const { return m_trig_log_file; }

  inline void setLedControlON(int x) { m_led_control_on = x; }
  inline int getLedControlON() const { return m_led_control_on; }

  inline void setLedControlHost(std::string x) { m_led_control_host = x; }
  inline std::string getLedControlHost() const { return m_led_control_host; }

  inline void setLedControlPort(int x) { m_led_control_port = x; }
  inline int getLedControlPort() const { return m_led_control_port; }

  inline void setIRLaserPower(int x) { m_ir_laser_power = x; }
  inline int getIRLaserPower() const { return m_ir_laser_power; }

  inline void setGreenLaserPower(int x) { m_green_laser_power = x; }
  inline int getGreenLaserPower() const { return m_green_laser_power; }

  inline void setRedLaserPower(int x) { 
    m_red_laser_power = x; 
    if (x != getBlue2LaserPower()) {
      setBlue2LaserPower(x);
    }
  }
  inline int getRedLaserPower() const { return m_red_laser_power; }
  // new in 2012
  inline void setBlue2LaserPower(int x) { 
    m_blue2_laser_power = x; 
    if (x != getRedLaserPower()) {
      setRedLaserPower(x);
    }
  }
  inline int getBlue2LaserPower() const { return m_blue2_laser_power; }
  //
  inline void setBlueLaserLogAttenuator(int x) { m_blue_laser_log_attenuator = x; }
  inline int getBlueLaserLogAttenuator() const { return m_blue_laser_log_attenuator; }

  inline void setIRLaserLogAttenuator(int x) { m_ir_laser_log_attenuator = x; }
  inline int getIRLaserLogAttenuator() const { return m_ir_laser_log_attenuator; }

  inline void setGreenLaserLogAttenuator(int x) { m_green_laser_log_attenuator = x; }
  inline int getGreenLaserLogAttenuator() const { return m_green_laser_log_attenuator; }

  inline void setRedLaserLogAttenuator(int x) { 
    m_red_laser_log_attenuator = x; 
    if (x != getBlue2LaserLogAttenuator()) {
      setBlue2LaserLogAttenuator(x);
    }
  }
  inline int getRedLaserLogAttenuator() const { return m_red_laser_log_attenuator; }
  // new in 2012
  inline void setBlue2LaserLogAttenuator(int x) { 
    m_blue2_laser_log_attenuator = x; 
    if (x != getRedLaserLogAttenuator()) {
      setRedLaserLogAttenuator(x);
    }
  }
  inline int getBlue2LaserLogAttenuator() const { return m_blue2_laser_log_attenuator; }
  //
  inline void setLaserConfigFile(std::string x) { m_laser_config_file = x; }
  inline std::string getLaserConfigFile() const { return m_laser_config_file ; }

  void setLaserClob(unsigned char *x, int size);
  inline std::vector<unsigned char> getLaserClob() const { return m_laser_clob; }
  std::string getLaserClobAsString() const;

  inline void printout() { 

    std::cout <<"Laser >>" <<"Size()                        " << getSize() <<std::endl << std::endl;
    std::cout <<"Laser >>" <<"Id()                          " << getId() <<std::endl;
    std::cout <<"Laser >>" <<"Tag()                         " << getConfigTag() <<std::endl;
    std::cout <<"Laser >>" <<"Debug()                       " << getDebug() <<std::endl;
    std::cout <<"Laser >>" <<"Dummy()                       " << getDummy() <<std::endl;
    std::cout <<"Laser >>" <<"MatacqBaseAddress()           " << getMatacqBaseAddress() <<std::endl;
    std::cout <<"Laser >>" <<"MatacqNone()                  " << getMatacqNone() <<std::endl;
    std::cout <<"Laser >>" <<"MatacqMode()                  " << getMatacqMode() <<std::endl; 
    std::cout <<"Laser >>" <<"ChannelMask()                 " << getChannelMask() <<std::endl;
    std::cout <<"Laser >>" <<"MaxSamplesForDaq()            " << getMaxSamplesForDaq() <<std::endl;
    std::cout <<"Laser >>" <<"MatacqFedId()                 " << getMatacqFedId() <<std::endl;    
    std::cout <<"Laser >>" <<"PedestalFile()                " << getPedestalFile() <<std::endl;
    std::cout <<"Laser >>" <<"UseBuffer()                   " << getUseBuffer() <<std::endl;
    std::cout <<"Laser >>" <<"[x] PostTrig()                " << getPostTrig() <<std::endl;
    std::cout <<"Laser >>" <<"[x] BlueLaserPostTrig()       " << getBlueLaserPostTrig() << std::endl;
    std::cout <<"Laser >>" <<"FPMode()                      " << getFPMode() <<std::endl;
    std::cout <<"Laser >>" <<"HalModuleFile()               " << getHalModuleFile() <<std::endl;
    std::cout <<"Laser >>" <<"HalAddressTableFile()         " << getHalAddressTableFile() <<std::endl;
    std::cout <<"Laser >>" <<"HalStaticTableFile()          " << getHalStaticTableFile() <<std::endl;
    std::cout <<"Laser >>" <<"MatacqSerialNumber()          " << getMatacqSerialNumber() <<std::endl; 
    std::cout <<"Laser >>" <<"PedestalRunEventCount()       " 
	      << getPedestalRunEventCount() <<std::endl;     
    std::cout <<"Laser >>" <<"RawDataMode()                 " << getRawDataMode() <<std::endl; 
    std::cout <<"Laser >>" <<"MatacqAcquisitionMode()       " 
	      << getMatacqAcquisitionMode() <<std::endl;     
    std::cout <<"Laser >>" <<"LocalOutputFile()             " << getLocalOutputFile() <<std::endl;     
    std::cout <<"Laser >>" <<"MatacqVernierMin()            " << getMatacqVernierMin() <<std::endl;  
    std::cout <<"Laser >>" <<"MatacqVernierMax()            " << getMatacqVernierMax() <<std::endl;    
    std::cout <<"Laser >>" <<"EMTCNone()                    " << getEMTCNone() <<std::endl; 
    std::cout <<"Laser >>" <<"[x] WTE2LaserDelay()          " << getWTE2LaserDelay() <<std::endl; 
    std::cout <<"Laser >>" <<"[x] WTE2BlueLaserDelay()      " << getWTE2BlueLaser() <<std::endl; 
    std::cout <<"Laser >>" <<"[n] WTE2IRLaserDelay()        " << getWTE2IRLaser() <<std::endl; 
    std::cout <<"Laser >>" <<"[n] WTE2Blue2LaserDelay()     " << getWTE2Blue2Laser() <<std::endl; 
    std::cout <<"Laser >>" <<"[n] WTE2GreenLaserDelay()     " << getWTE2GreenLaser() <<std::endl; 
    std::cout <<"Laser >>" <<"[x] LaserPhase()              " << getLaserPhase() <<std::endl;      
    std::cout <<"Laser >>" <<"[x] BlueLaserPhase()          " << getBlueLaserPhase() <<std::endl;      
    std::cout <<"Laser >>" <<"[n] IRLaserPhase()            " << getIRLaserPhase() <<std::endl;      
    std::cout <<"Laser >>" <<"[n] Blue2LaserPhase()         " << getBlue2LaserPhase() <<std::endl; 
    std::cout <<"Laser >>" <<"EMTCTTCIn()                   " << getEMTCTTCIn() <<std::endl;  
    std::cout <<"Laser >>" <<"EMTCSlotId()                  " << getEMTCSlotId() <<std::endl; 
    std::cout <<"Laser >>" <<"WaveLength()                  " << getWaveLength() <<std::endl; 
    std::cout <<"Laser >>" <<"Power()                       " << getPower() <<std::endl;       
    std::cout <<"Laser >>" <<"[n] BlueLaserPower()          " << getBlueLaserPower() <<std::endl;       
    std::cout <<"Laser >>" <<"OpticalSwitch()               " << getOpticalSwitch() <<std::endl;  
    std::cout <<"Laser >>" <<"Filter()                      " << getFilter() <<std::endl;       
    std::cout <<"Laser >>" <<"LaserControlOn()              " << getLaserControlOn() <<std::endl; 
    std::cout <<"Laser >>" <<"LaserControlHost()            " << getLaserControlHost() <<std::endl; 
    std::cout <<"Laser >>" <<"LaserControlPort()            " << getLaserControlPort() <<std::endl; 
    std::cout <<"Laser >>" <<"LaserTag()                    " << getLaserTag() <<std::endl;  
    std::cout <<"Laser >>" <<"WTE2LedDelay()                " << getWTE2LedDelay() <<std::endl; 
    std::cout <<"Laser >>" <<"[n] WTE2LedSoakDelay()        " << getWTE2LedSoakDelay() <<std::endl; 
    std::cout <<"Laser >>" <<"[n] LedPostScale()            " << getLedPostScale() <<std::endl; 
    std::cout <<"Laser >>" <<"Led1ON()                      " << getLed1ON() <<std::endl;        
    std::cout <<"Laser >>" <<"Led2ON()                      " << getLed2ON() <<std::endl;
    std::cout <<"Laser >>" <<"Led3ON()                      " << getLed3ON() <<std::endl; 
    std::cout <<"Laser >>" <<"Led4ON()                      " << getLed4ON() <<std::endl;  
    std::cout <<"Laser >>" <<"Vinj()                        " << getVinj() <<std::endl; 
    std::cout <<"Laser >>" <<"OrangeLedMonAmpl()            " << getOrangeLedMonAmpl() <<std::endl; 
    std::cout <<"Laser >>" <<"BlueLedMonAmpl()              " << getBlueLedMonAmpl() <<std::endl;    
    std::cout <<"Laser >>" <<"TrigLogFile()                 " << getTrigLogFile() <<std::endl;
    std::cout <<"Laser >>" <<"LedControlON()                " << getLedControlON() <<std::endl;
    std::cout <<"Laser >>" <<"LedControlHost()              " << getLedControlHost() <<std::endl;
    std::cout <<"Laser >>" <<"LedControlPort()              " << getLedControlPort() <<std::endl;
    std::cout <<"Laser >>" <<"IRLaserPower()                " << getIRLaserPower() <<std::endl;
    std::cout <<"Laser >>" <<"GreenLaserPower()             " << getGreenLaserPower() <<std::endl;
    std::cout <<"Laser >>" <<"[x] RedLaserPower()           " << getRedLaserPower() <<std::endl;
    std::cout <<"Laser >>" <<"[x] Blue2LaserPower()         " << getBlue2LaserPower() <<std::endl;
    std::cout <<"Laser >>" <<"BlueLaserLogAttenuator()      " << getBlueLaserLogAttenuator() 
	      <<std::endl;    
    std::cout <<"Laser >>" <<"IRLaserLogAttenuator()        " << getIRLaserLogAttenuator() <<std::endl;
    std::cout <<"Laser >>" <<"GreenLaserLogAttenuator()     " << getGreenLaserLogAttenuator() 
	      <<std::endl;     
    std::cout <<"Laser >>" <<"[x] RedLaserLogAttenuator()   " << getRedLaserLogAttenuator() 
	      <<std::endl;     
    std::cout <<"Laser >>" <<"[x] Blue2LaserLogAttenuator() " << getBlue2LaserLogAttenuator() 
	      <<std::endl;     
    std::cout <<"Laser >>" <<"LaserConfigFile()             " << getLaserConfigFile() <<std::endl; 
    std::cout <<"Laser >>" <<"LaserClob()                   " << getLaserClobAsString() <<std::endl;
  }

  void setParameters(std::map<std::string,std::string> my_keys_map);

  int fetchNextId() throw(std::runtime_error);
  
 private:
  std::string values(int i);
  bool setDB(); // determines if we are reading a pre-2012 db
  void prepareWrite()  throw(std::runtime_error);
  void writeDB()       throw(std::runtime_error);
  void clear();
  void fetchData(ODLaserConfig * result)     throw(std::runtime_error);
  int fetchID()  throw(std::runtime_error);

  // internal data
  bool m_isOldDb;
  bool m_db_checked;

  // User data
  int m_ID;

  // mataq
  int m_dummy;
  int m_debug;
  int m_mq_base;
  int m_mq_none;
  std::string m_mode ;
  int m_chan_mask;
  std::string m_samples;
  int m_mq_fed;  
  std::string m_ped_file;
  int  m_use_buffer;
  int  m_post_trig;
  // new in 2012
  int m_blue_laser_post_trig; //replace POSTTRIG
  int m_blue2_laser_post_trig;
  int m_ir_laser_post_trig;
  int m_green_laser_post_trig;
  //
  int  m_fp_mode;
  std::string m_hal_mod_file;
  std::string m_hal_add_file;
  std::string m_hal_tab_file;
  std::string m_serial;
  int  m_ped_count;
  int  m_raw_mode;
  std::string m_aqmode;
  std::string m_mq_file;
  int m_matacq_vernier_min;
  int m_matacq_vernier_max;

  // emtc 
  int m_emtc_1;
  int m_emtc_2;
  int m_emtc_3;
  // new in 2012
  int m_green_laser_phase;
  int m_ir_laser_phase;
  int m_blue_laser_phase; // replace LASER_PHASE
  int m_blue2_laser_phase;
  //
  int m_emtc_4;
  int m_emtc_5;

  // laser
  int m_wave;
  int m_power;
  // new in 2012
  int m_blue_laser_power;
  //
  int m_switch;
  int m_filter;
  int m_on;
  std::string m_laserhost;
  int m_laserport;

  std::string m_laser_tag;


  // led 
 int m_wte_2_led_delay;
 // new in 2012
 int m_wte2_blue_laser; // replace WTE2_LASER_DELAY
 int m_wte2_blue2_laser;
 int m_wte2_ir_laser;
 int m_wte2_green_laser;

 int m_wte_2_led_soak_delay;
 int m_led_postscale;
 //
 int m_led1_on; 
 int m_led2_on ;
 int m_led3_on ;
 int m_led4_on ;
 int m_vinj;
 int m_orange_led_mon_ampl ;
 int m_blue_led_mon_ampl ;
 std::string m_trig_log_file; 
 int m_led_control_on ;
 std::string m_led_control_host; 
 int m_led_control_port ;
 int m_ir_laser_power ;
 int m_green_laser_power; 
 int m_red_laser_power ;
 int m_blue_laser_log_attenuator; 
 // new in 2012
 int m_blue2_laser_power; // replace red_laser_power
 int m_blue2_laser_log_attenuator; // red_laser_log_attenuator
 //
 int m_ir_laser_log_attenuator;
 int m_green_laser_log_attenuator;
 int m_red_laser_log_attenuator;
 std::string m_laser_config_file;
 std::vector<unsigned char> m_laser_clob ;
 unsigned int m_size;


};

#endif
