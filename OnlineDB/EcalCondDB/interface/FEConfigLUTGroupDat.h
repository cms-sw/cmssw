#ifndef FECONFLUTGROUPDAT_H
#define FECONFLUTGROUPDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLUTInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigLUTGroupDat : public IDataItem {
public:
  friend class EcalCondDBInterface;  // XXX temp should not need
  FEConfigLUTGroupDat();
  ~FEConfigLUTGroupDat() override;

  // User data methods
  inline std::string getTable() override { return "FE_CONFIG_LUT_PER_GROUP_DAT"; }

  inline void setLUTGroupId(int x) { m_group_id = x; }
  inline int getLUTGroupId() const { return m_group_id; }

  inline void setLUTValue(int i, int x) { m_lut[i] = x; }
  inline int getLUTValue(int i) const { return m_lut[i]; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const FEConfigLUTGroupDat* item, FEConfigLUTInfo* iconf) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, FEConfigLUTGroupDat>* data, FEConfigLUTInfo* iconf) noexcept(false);

  void fetchData(std::map<EcalLogicID, FEConfigLUTGroupDat>* fillMap, FEConfigLUTInfo* iconf) noexcept(false);

  // User data
  int m_group_id;
  int m_lut[1024];
};

#endif
