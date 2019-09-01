#ifndef FECONFFGRDAT_H
#define FECONFFGRDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigFgrInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigFgrDat : public IDataItem {
public:
  friend class EcalCondDBInterface;  // XXX temp should not need
  FEConfigFgrDat();
  ~FEConfigFgrDat() override;

  // User data methods
  inline std::string getTable() override { return "FE_CONFIG_FGR_DAT"; }

  inline void setFgrGroupId(int x) { m_group_id = x; }
  inline int getFgrGroupId() const { return m_group_id; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const FEConfigFgrDat* item, FEConfigFgrInfo* iconf) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, FEConfigFgrDat>* data, FEConfigFgrInfo* iconf) noexcept(false);

  void fetchData(std::map<EcalLogicID, FEConfigFgrDat>* fillMap, FEConfigFgrInfo* iconf) noexcept(false);

  // User data
  int m_group_id;
};

#endif
