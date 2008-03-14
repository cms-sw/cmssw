#ifndef ODMATAQCONFIG_H
#define ODMATAQCONFIG_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"

class ODMataqConfig : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  ODMataqConfig();
  ~ODMataqConfig();

  // User data methods
  inline std::string getTable() { return "ECAL_Mataq_CONFIGURATION"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  inline void setMataqMode(std::string x) { m_mode = x; }
  inline std::string getMataqMode() const { return m_mode; }

  inline void setFastPedestal(int x) { m_fast_ped = x; }
  inline int getFastPedestal() const { return m_fast_ped; }

  inline void setChannelMask(int x) { m_chan_mask = x; }
  inline int getChannelMask() const { return m_chan_mask; }

  inline void setMaxSamplesForDaq(std::string x) { m_samples = x; }
  inline std::string getMaxSamplesForDaq() const { return m_samples; }

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

  inline void setMataqSerialNumber(std::string x) { m_serial = x; }
  inline std::string getMataqSerialNumber() const { return m_serial; }

  inline void setPedestalRunEventCount(int x) { m_ped_count = x; }
  inline int getPedestalRunEventCount() const { return m_ped_count; }

  inline void setRawDataMode(int x) { m_raw_mode = x; }
  inline int getRawDataMode() const { return m_raw_mode; }

  
 private:
  void prepareWrite()  throw(std::runtime_error);
  void writeDB()       throw(std::runtime_error);
  void clear();
  void fetchData(ODMataqConfig * result)     throw(std::runtime_error);
  int fetchID()  throw(std::runtime_error);


  // User data
  int m_ID;

  std::string m_mode ;
  int m_fast_ped ;
  int m_chan_mask;
  std::string m_samples;
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

};

#endif
