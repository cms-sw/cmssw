#ifndef ODSCANCYCLE_H
#define ODSCANCYCLE_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"

class ODScanCycle : public IODConfig {
public:
  friend class EcalCondDBInterface;
  friend class ODEcalCycle;

  ODScanCycle();
  ~ODScanCycle() override;

  inline std::string getTable() override { return "ECAL_Scan_CYCLE"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; };

  // Methods for user data
  inline void setScanConfigurationID(int x) { m_scan_config_id = x; }
  inline int getScanConfigurationID() const { return m_scan_config_id; }

  // Operators
  inline bool operator==(const ODScanCycle &m) const { return (m_ID == m.m_ID); }
  inline bool operator!=(const ODScanCycle &m) const { return !(*this == m); }

private:
  // User data
  int m_ID;
  int m_scan_config_id;
  void writeDB() noexcept(false);
  void prepareWrite() noexcept(false) override;
  void clear();
  void fetchData(ODScanCycle *result) noexcept(false);
  void insertConfig() noexcept(false);

  // Methods from IUniqueDBObject
  int fetchID() noexcept(false);
  void setByID(int id) noexcept(false);
};

#endif
