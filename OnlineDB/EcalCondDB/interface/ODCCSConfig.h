#ifndef ODCCSCONFIG_H
#define ODCCSCONFIG_H

#include <map>
#include <string>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"

class ODCCSConfig : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  ODCCSConfig();
  ~ODCCSConfig();

  // User data methods
  inline std::string getTable() { return "ECAL_CCS_CONFIGURATION"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  inline void setDaccal(int x) { m_daccal = x; }
  inline int getDaccal() const { return m_daccal; }

  inline void setDelay(int x) { m_delay = x; }
  inline int getDelay() const { return m_delay; }

  inline void setGain(std::string x) { m_gain = x; }
  inline std::string getGain() const { return m_gain; }

  inline void setMemGain(std::string x) { m_memgain = x; }
  inline std::string getMemGain() const { return m_memgain; }

  inline void setOffsetHigh(int x) { m_offset_high = x; }
  inline int getOffsetHigh() const { return m_offset_high; }

  inline void setOffsetLow(int x) { m_offset_low = x; }
  inline int getOffsetLow() const { return m_offset_low; }

  inline void setOffsetMid(int x) { m_offset_mid = x; }
  inline int getOffsetMid() const { return m_offset_mid; }

  inline void setTrgMode(std::string x) { m_trg_mode = x; }
  inline std::string getTrgMode() const { return m_trg_mode; }

  inline void setTrgFilter(std::string x) { m_trg_filter = x; }
  inline std::string getTrgFilter() const { return m_trg_filter; }

  inline void setClock(int x) { m_clock = x; }
  inline int getClock() const { return m_clock; }
  inline void setBGOSource(std::string x) { m_bgo = x; }
  inline std::string getBGOSource() const { return m_bgo; }
  inline void setTTSMask(int x) { m_tts_mask = x; }
  inline int getTTSMask() const { return m_tts_mask; }
  inline void setDAQBCIDPreset(int x) { m_daq = x; }
  inline int getDAQBCIDPreset() const { return m_daq; }
  inline void setTrgBCIDPreset(int x) { m_trg = x; }
  inline int getTrgBCIDPreset() const { return m_trg; }
  inline void setBC0Counter(int x) { m_bc0 = x; }
  inline int getBC0Counter() const { return m_bc0; }
  inline int getBC0Delay() const { return m_bc0_delay; }
  inline void setBC0Delay(int x) { m_bc0_delay = x; }
  inline int getTEDelay() const { return m_te_delay; }
  inline void setTEDelay(int x) { m_te_delay = x; }


  int fetchNextId() throw(std::runtime_error);
  void setParameters(std::map<std::string,std::string> my_keys_map);
  
 private:
  void prepareWrite()  throw(std::runtime_error);

  void writeDB()       throw(std::runtime_error);

  void clear();

  void fetchData(ODCCSConfig * result)     throw(std::runtime_error);

  int fetchID()  throw(std::runtime_error);


  // User data
  int m_ID;
  int m_daccal;
  int m_delay;
  std::string m_gain;
  std::string m_memgain;
  int m_offset_high;
  int m_offset_low;
  int m_offset_mid;
  std::string m_pedestal_offset_release;
  std::string m_system;
  std::string m_trg_mode;
  std::string m_trg_filter;
  int m_clock;
  std::string m_bgo;
  int m_tts_mask;
  int m_daq;
  int m_trg;
  int m_bc0;
  int m_bc0_delay;
  int m_te_delay;
  
};

#endif
