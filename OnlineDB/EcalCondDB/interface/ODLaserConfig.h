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

  inline void setWaveLength(int x) { m_wave = x; }
  inline int getWaveLength() const { return m_wave; }

  inline void setPower(int x) { m_power = x; }
  inline int getPower() const { return m_power; }

  inline void setOpticalSwitch(int x) { m_switch = x; }
  inline int getOpticalSwitch() const { return m_switch; }

  inline void setFilter(int x) { m_filter = x; }
  inline int getFilter() const { return m_filter; }
  
 private:
  void prepareWrite()  throw(std::runtime_error);
  void writeDB()       throw(std::runtime_error);
  void clear();
  void fetchData(ODLaserConfig * result)     throw(std::runtime_error);
  int fetchID()  throw(std::runtime_error);


  // User data
  int m_ID;
  int m_wave;
  int m_power;
  int m_switch;
  int m_filter;
};

#endif
