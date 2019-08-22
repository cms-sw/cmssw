#ifndef FECONFWEIGHTDAT_H
#define FECONFWEIGHTDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigWeightInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigWeightDat : public IDataItem {
public:
  friend class EcalCondDBInterface;  // XXX temp should not need
  FEConfigWeightDat();
  ~FEConfigWeightDat() override;

  // User data methods
  inline std::string getTable() override { return "FE_CONFIG_WEIGHT_DAT"; }

  inline void setWeightGroupId(int x) { m_group_id = x; }
  inline int getWeightGroupId() const { return m_group_id; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const FEConfigWeightDat* item, FEConfigWeightInfo* iconf) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, FEConfigWeightDat>* data, FEConfigWeightInfo* iconf) noexcept(false);

  void fetchData(std::map<EcalLogicID, FEConfigWeightDat>* fillMap, FEConfigWeightInfo* iconf) noexcept(false);

  // User data
  int m_group_id;
};

#endif
