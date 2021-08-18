#ifndef ONLINEDB_ECALCONDDB_FECONFIGODDWEIGHTDAT
#define ONLINEDB_ECALCONDDB_FECONFIGODDWEIGHTDAT

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigOddWeightInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigOddWeightDat : public IDataItem {
public:
  friend class EcalCondDBInterface;  // XXX temp should not need
  FEConfigOddWeightDat();
  ~FEConfigOddWeightDat() override;

  // User data methods
  inline std::string getTable() override { return "FE_CONFIG_WEIGHT2_DAT"; }

  inline void setWeightGroupId(int x) { m_group_id = x; }
  inline int getWeightGroupId() const { return m_group_id; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const FEConfigOddWeightDat* item, FEConfigOddWeightInfo* iconf) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, FEConfigOddWeightDat>* data,
                    FEConfigOddWeightInfo* iconf) noexcept(false);

  void fetchData(std::map<EcalLogicID, FEConfigOddWeightDat>* fillMap, FEConfigOddWeightInfo* iconf) noexcept(false);

  // User data
  int m_group_id;
};

#endif
