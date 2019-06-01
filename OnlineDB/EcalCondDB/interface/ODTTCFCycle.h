#ifndef ODTTCFCYCLE_H
#define ODTTCFCYCLE_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"

class ODTTCFCycle : public IODConfig {
public:
  friend class EcalCondDBInterface;
  friend class ODEcalCycle;

  ODTTCFCycle();
  ~ODTTCFCycle() override;

  inline std::string getTable() override { return "ECAL_TTCF_CYCLE"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; };

  // Methods for user data
  inline void setTTCFConfigurationID(int x) { m_ttcf_config_id = x; }
  inline int getTTCFConfigurationID() const { return m_ttcf_config_id; }

  // Operators
  inline bool operator==(const ODTTCFCycle &m) const { return (m_ID == m.m_ID); }
  inline bool operator!=(const ODTTCFCycle &m) const { return !(*this == m); }

private:
  // User data
  int m_ID;
  int m_ttcf_config_id;
  void writeDB() noexcept(false);
  void prepareWrite() noexcept(false) override;
  void clear();
  void fetchData(ODTTCFCycle *result) noexcept(false);
  void insertConfig() noexcept(false);

  // Methods from IUniqueDBObject
  int fetchID() noexcept(false);
  void setByID(int id) noexcept(false);
};

#endif
