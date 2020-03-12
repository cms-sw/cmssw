#ifndef FECONFIGTIMING_H
#define FECONFIGTIMING_H

#include <map>
#include <string>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

class FEConfigTimingInfo : public IODConfig {
public:
  friend class EcalCondDBInterface;

  FEConfigTimingInfo();
  ~FEConfigTimingInfo() override;

  // Methods for user data
  inline std::string getTable() override { return "FE_CONFIG_TIME_INFO"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }
  // for compatibility
  void setID(int id) { setId(id); }
  int getID() { return getId(); }
  // the tag is already in IODConfig
  inline void setVersion(int id) { m_version = id; }
  inline int getVersion() const { return m_version; }

  Tm getDBTime() const { return m_db_time; }
  void setDBTime(const Tm& x) { m_db_time = x; }

  // Methods from IUniqueDBObject
  int fetchID() noexcept(false);
  int fetchNextId() noexcept(false);
  void setByID(int id) noexcept(false);
  void setParameters(const std::map<std::string, std::string>& my_keys_map);

  // operators
  inline bool operator==(const FEConfigTimingInfo& r) const { return (m_ID == r.m_ID); }
  inline bool operator!=(const FEConfigTimingInfo& r) const { return !(*this == r); }

private:
  // User data for this IOV
  int m_ID;
  Tm m_db_time;
  int m_version;

  void prepareWrite() noexcept(false) override;
  void writeDB() noexcept(false);
  void clear();
  void fetchData(FEConfigTimingInfo* result) noexcept(false);
  void fetchLastData(FEConfigTimingInfo* result) noexcept(false);
};

#endif
