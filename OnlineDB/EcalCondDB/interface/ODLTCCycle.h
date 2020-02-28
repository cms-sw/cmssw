#ifndef ODLTCCYCLE_H
#define ODLTCCYCLE_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"

class ODLTCCycle : public IODConfig {
public:
  friend class EcalCondDBInterface;
  friend class ODEcalCycle;

  ODLTCCycle();
  ~ODLTCCycle() override;

  inline std::string getTable() override { return "ECAL_LTC_CYCLE"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; };

  // Methods for user data
  inline void setLTCConfigurationID(int x) { m_ltc_config_id = x; }
  inline int getLTCConfigurationID() const { return m_ltc_config_id; }

  // Operators
  inline bool operator==(const ODLTCCycle &m) const { return (m_ID == m.m_ID); }
  inline bool operator!=(const ODLTCCycle &m) const { return !(*this == m); }

private:
  // User data
  int m_ID;
  int m_ltc_config_id;
  void writeDB() noexcept(false);
  void prepareWrite() noexcept(false) override;
  void clear();
  void fetchData(ODLTCCycle *result) noexcept(false);
  void insertConfig() noexcept(false);

  // Methods from IUniqueDBObject
  int fetchID() noexcept(false);
  void setByID(int id) noexcept(false);
};

#endif
