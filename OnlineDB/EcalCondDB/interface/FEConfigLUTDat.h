#ifndef FECONFLUTDAT_H
#define FECONFLUTDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLUTInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigLUTDat : public IDataItem {
public:
  friend class EcalCondDBInterface;  // XXX temp should not need
  FEConfigLUTDat();
  ~FEConfigLUTDat() override;

  // User data methods
  inline std::string getTable() override { return "FE_CONFIG_LUT_DAT"; }

  inline void setLUTGroupId(int x) { m_group_id = x; }
  inline int getLUTGroupId() const { return m_group_id; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const FEConfigLUTDat* item, FEConfigLUTInfo* iconf) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, FEConfigLUTDat>* data, FEConfigLUTInfo* iconf) noexcept(false);

  void fetchData(std::map<EcalLogicID, FEConfigLUTDat>* fillMap, FEConfigLUTInfo* iconf) noexcept(false);

  // User data
  int m_group_id;
};

#endif
