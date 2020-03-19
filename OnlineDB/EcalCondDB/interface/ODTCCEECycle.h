#ifndef ODTCCEECYCLE_H
#define ODTCCEECYCLE_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"

class ODTCCEECycle : public IODConfig {
public:
  friend class EcalCondDBInterface;
  friend class ODEcalCycle;

  ODTCCEECycle();
  ~ODTCCEECycle() override;

  inline std::string getTable() override { return "ECAL_TCC_EE_CYCLE"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; };

  // Methods for user data
  inline void setTCCConfigurationID(int x) { m_tcc_ee_config_id = x; }
  inline int getTCCConfigurationID() const { return m_tcc_ee_config_id; }

  // Operators
  inline bool operator==(const ODTCCEECycle &m) const { return (m_ID == m.m_ID); }
  inline bool operator!=(const ODTCCEECycle &m) const { return !(*this == m); }

private:
  // User data
  int m_ID;
  int m_tcc_ee_config_id;
  void writeDB() noexcept(false);
  void prepareWrite() noexcept(false) override;
  void clear();
  void fetchData(ODTCCEECycle *result) noexcept(false);
  void insertConfig() noexcept(false);

  // Methods from IUniqueDBObject
  int fetchID() noexcept(false);
  void setByID(int id) noexcept(false);
};

#endif
