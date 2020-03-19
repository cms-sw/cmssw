#ifndef ODSRPCYCLE_H
#define ODSRPCYCLE_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"

class ODSRPCycle : public IODConfig {
public:
  friend class EcalCondDBInterface;
  friend class ODEcalCycle;

  ODSRPCycle();
  ~ODSRPCycle() override;

  inline std::string getTable() override { return "ECAL_SRP_CYCLE"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; };

  // Methods for user data
  inline void setSRPConfigurationID(int x) { m_srp_config_id = x; }
  inline int getSRPConfigurationID() const { return m_srp_config_id; }

  // Operators
  inline bool operator==(const ODSRPCycle &m) const { return (m_ID == m.m_ID); }
  inline bool operator!=(const ODSRPCycle &m) const { return !(*this == m); }

private:
  // User data
  int m_ID;
  int m_srp_config_id;
  void writeDB() noexcept(false);
  void prepareWrite() noexcept(false) override;
  void clear();
  void fetchData(ODSRPCycle *result) noexcept(false);
  void insertConfig() noexcept(false);

  // Methods from IUniqueDBObject
  int fetchID() noexcept(false);
  void setByID(int id) noexcept(false);
};

#endif
