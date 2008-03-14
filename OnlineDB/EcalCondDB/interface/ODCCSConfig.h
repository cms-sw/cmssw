#ifndef ODCCSCONFIG_H
#define ODCCSCONFIG_H

#include <map>
#include <stdexcept>

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

  inline void setGain(int x) { m_gain = x; }
  inline int getGain() const { return m_gain; }

  inline void setMemGain(int x) { m_memgain = x; }
  inline int getMemGain() const { return m_memgain; }

  inline void setOffsetHigh(int x) { m_offset_high = x; }
  inline int getOffsetHigh() const { return m_offset_high; }

  inline void setOffsetLow(int x) { m_offset_low = x; }
  inline int getOffsetLow() const { return m_offset_low; }

  inline void setOffsetMid(int x) { m_offset_mid = x; }
  inline int getOffsetMid() const { return m_offset_mid; }

  inline void setPedestalOffsetRelease(std::string x) { m_pedestal_offset_release = x; }
  inline std::string getPedestalOffsetRelease() const { return m_pedestal_offset_release; }

  inline void setSystem(std::string x) { m_system = x; }
  inline std::string getSystem() const { return m_system; }

  inline void setTrgMode(std::string x) { m_trg_mode = x; }
  inline std::string getTrgMode() const { return m_trg_mode; }

  inline void setTrgFilter(std::string x) { m_trg_filter = x; }
  inline std::string getTrgFilter() const { return m_trg_filter; }

  
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
  int m_gain;
  int m_memgain;
  int m_offset_high;
  int m_offset_low;
  int m_offset_mid;
  std::string m_pedestal_offset_release;
  std::string m_system;
  std::string m_trg_mode;
  std::string m_trg_filter;
  
};

#endif
