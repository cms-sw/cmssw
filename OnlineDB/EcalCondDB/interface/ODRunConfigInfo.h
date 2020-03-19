#ifndef ODRUNCONFIGINFO_H
#define ODRUNCONFIGINFO_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/RunModeDef.h"
#include "OnlineDB/EcalCondDB/interface/RunTypeDef.h"

class ODRunConfigInfo : public IODConfig {
public:
  friend class EcalCondDBInterface;

  ODRunConfigInfo();
  ~ODRunConfigInfo() override;
  inline std::string getTable() override { return "ECAL_RUN_CONFIGURATION_DAT"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  void setDBTime(const Tm& x) { m_db_time = x; }
  inline Tm getDBTime() const { return m_db_time; }
  //
  inline void setTag(std::string x) { m_tag = x; }
  std::string getTag() const { return m_tag; }
  //
  void setDescription(std::string x) { m_description = x; }
  std::string getDescription() const { return m_description; }
  //
  void setVersion(int x) { m_version = x; }
  int getVersion() const { return m_version; }
  //
  void setNumberOfSequences(int n) { m_num_seq = n; }
  int getNumberOfSequences() const { return m_num_seq; }
  //
  void setDefaults(int x) { m_defaults = x; }
  int getDefaults() const { return m_defaults; }
  //
  void setTriggerMode(std::string x) { m_trigger_mode = x; }
  std::string getTriggerMode() const { return m_trigger_mode; }
  //
  void setNumberOfEvents(int x) { m_num_events = x; }
  int getNumberOfEvents() const { return m_num_events; }
  //
  void setUsageStatus(std::string x) { m_usage_status = x; }
  std::string getUsageStatus() const { return m_usage_status; }
  //

  RunTypeDef getRunTypeDef() const;
  void setRunTypeDef(const RunTypeDef& runTypeDef);
  RunModeDef getRunModeDef() const;
  void setRunModeDef(const RunModeDef& runModeDef);

  // operators
  inline bool operator==(const ODRunConfigInfo& r) const { return (m_ID == r.m_ID); }
  inline bool operator!=(const ODRunConfigInfo& r) const { return !(*this == r); }

private:
  // User data for this IOV
  int m_ID;
  Tm m_db_time;
  std::string m_tag;
  int m_version;
  RunModeDef m_runModeDef;
  RunTypeDef m_runTypeDef;
  int m_num_seq;
  std::string m_description;
  int m_defaults;
  std::string m_trigger_mode;
  int m_num_events;
  std::string m_usage_status;

  // Methods from IUniqueDBObject
  int fetchNextId() noexcept(false);
  int fetchID() noexcept(false);
  int fetchIDFromTagAndVersion() noexcept(false);
  int fetchIDLast() noexcept(false);
  void setByID(int id) noexcept(false);

  void prepareWrite() noexcept(false) override;
  void writeDB() noexcept(false);
  void fetchData(ODRunConfigInfo* result) noexcept(false);
  int updateDefaultCycle() noexcept(false);
  void clear();
};

#endif
