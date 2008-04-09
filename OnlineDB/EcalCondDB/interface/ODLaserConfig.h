#ifndef ODLASERCONFIG_H
#define ODLASERCONFIG_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"

class ODLaserConfig : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  ODLaserConfig();
  ~ODLaserConfig();

  // User data methods
  inline std::string getTable() { return "ECAL_Laser_CONFIGURATION"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  inline void setDebug(int x) { m_debug = x; }
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
  inline void setPostTrig(int x) { m_post_trig = x; }
  inline int getPostTrig() const { return m_post_trig; }
  inline void setFPMode(int x) { m_fp_mode = x; }
  inline int getFPMode() const { return m_fp_mode; }
  inline void setHalModuleFile(std::string x) { m_hal_mod_file = x; }
  inline std::string getHalModuleFile() const { return m_hal_mod_file; }

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
  inline void setWTE2LaserDelay(int x) { m_emtc_2 = x; }
  inline int getWTE2LaserDelay() const { return m_emtc_2; }
  inline void setLaserPhase(int x) { m_emtc_3 = x; }
  inline int getLaserPhase() const { return m_emtc_3; }
  inline void setEMTCTTCIn(int x) { m_emtc_4 = x; }
  inline int getEMTCTTCIn() const { return m_emtc_4; }
  inline void setEMTCSlotId(int x) { m_emtc_5 = x; }
  inline int getEMTCSlotId() const { return m_emtc_5; }



  // laser 

  inline void setWaveLength(int x) { m_wave = x; }
  inline int getWaveLength() const { return m_wave; }

  inline void setPower(int x) { m_power = x; }
  inline int getPower() const { return m_power; }

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

  int fetchNextId() throw(std::runtime_error);
  
 private:
  void prepareWrite()  throw(std::runtime_error);
  void writeDB()       throw(std::runtime_error);
  void clear();
  void fetchData(ODLaserConfig * result)     throw(std::runtime_error);
  int fetchID()  throw(std::runtime_error);


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
  int  m_fp_mode;
  std::string m_hal_mod_file;
  std::string m_hal_add_file;
  std::string m_hal_tab_file;
  std::string m_serial;
  int  m_ped_count;
  int  m_raw_mode;
  std::string m_aqmode;
  std::string m_mq_file;

  // emtc 
  int m_emtc_1;
  int m_emtc_2;
  int m_emtc_3;
  int m_emtc_4;
  int m_emtc_5;

  // laser
  int m_wave;
  int m_power;
  int m_switch;
  int m_filter;
  int m_on;
  std::string m_laserhost;
  int m_laserport;

  std::string m_laser_tag;

};

#endif
