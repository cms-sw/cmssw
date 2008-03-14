#ifndef ODLTCCONFIG_H
#define ODLTCCONFIG_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"

class ODLTCConfig : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  ODLTCConfig();
  ~ODLTCConfig();

  // User data methods
  inline std::string getTable() { return "ECAL_LTC_CONFIGURATION"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  inline void setDeviceConfigParamId(int x) { m_dev = x; }
  inline int getDeviceConfigParamId() const { return m_dev; }
  
 private:
  void prepareWrite()  throw(std::runtime_error);

  void writeDB()       throw(std::runtime_error);

  void clear();

  void fetchData(ODLTCConfig * result)     throw(std::runtime_error);

  int fetchID()  throw(std::runtime_error);


  // User data
  int m_ID;
  int m_dev;
  
};

#endif
