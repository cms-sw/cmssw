#ifndef ODRUNCONFIGCYCLEINFO_H
#define ODRUNCONFIGCYCLEINFO_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/RunModeDef.h"
#include "OnlineDB/EcalCondDB/interface/RunTypeDef.h"
#include "OnlineDB/EcalCondDB/interface/RunSeqDef.h"
#include "OnlineDB/EcalCondDB/interface/IODConfig.h"

class ODRunConfigCycleInfo : public IODConfig {
public:
  friend class EcalCondDBInterface;
  friend class ODEcalCycle;

  ODRunConfigCycleInfo();
  ~ODRunConfigCycleInfo() override;

  inline std::string getTable() override { return "ECAL_CYCLE_DAT"; }

  // Methods for user data
  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  void setDescription(std::string x) { m_description = x; }
  std::string getDescription() const { return m_description; }
  //
  void setTag(std::string x) { m_tag = x; }
  std::string getTag() const { return m_tag; }
  //
  void setSequenceID(int x) { m_sequence_id = x; }
  int getSequenceID() const { return m_sequence_id; }
  //
  void setCycleNumber(int n) { m_cycle_num = n; }
  int getCycleNumber() const { return m_cycle_num; }
  //

  // operators
  inline bool operator==(const ODRunConfigCycleInfo &r) const { return (m_ID == r.m_ID); }
  inline bool operator!=(const ODRunConfigCycleInfo &r) const { return !(*this == r); }

private:
  // User data for this IOV
  int m_ID;
  int m_sequence_id;
  int m_cycle_num;
  std::string m_tag;
  std::string m_description;

  void prepareWrite() noexcept(false) override;
  void writeDB() noexcept(false);
  void clear();
  void fetchData(ODRunConfigCycleInfo *result) noexcept(false);
  void insertConfig() noexcept(false);

  // Methods from IUniqueDBObject
  int fetchID() noexcept(false);      // fetches the Cycle by the seq_id and cycle_num
  int fetchIDLast() noexcept(false);  // fetches the Cycle by the seq_id and cycle_num
  void setByID(int id) noexcept(false);
};

#endif
