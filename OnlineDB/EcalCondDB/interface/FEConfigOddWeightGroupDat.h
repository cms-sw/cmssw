#ifndef ONLINEDB_ECALCONDDB_FECONFIGODDWEIGHTGROUPDAT_H
#define ONLINEDB_ECALCONDDB_FECONFIGODDWEIGHTGROUPDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigOddWeightInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigOddWeightGroupDat : public IDataItem {
public:
  friend class EcalCondDBInterface;  // XXX temp should not need
  FEConfigOddWeightGroupDat();
  ~FEConfigOddWeightGroupDat() override;

  // User data methods
  inline std::string getTable() override { return "FE_WEIGHT2_PER_GROUP_DAT"; }

  inline void setWeightGroupId(int x) { m_group_id = x; }
  inline int getWeightGroupId() const { return m_group_id; }

  inline void setWeight0(float x) { m_w0 = x; }
  inline float getWeight0() const { return m_w0; }
  inline void setWeight1(float x) { m_w1 = x; }
  inline float getWeight1() const { return m_w1; }
  inline void setWeight2(float x) { m_w2 = x; }
  inline float getWeight2() const { return m_w2; }
  inline void setWeight3(float x) { m_w3 = x; }
  inline float getWeight3() const { return m_w3; }
  inline void setWeight4(float x) { m_w4 = x; }
  inline float getWeight4() const { return m_w4; }
  inline void setWeight5(float x) { m_w5 = x; }
  inline float getWeight5() const { return m_w5; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid,
               const FEConfigOddWeightGroupDat* item,
               FEConfigOddWeightInfo* iconf) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, FEConfigOddWeightGroupDat>* data,
                    FEConfigOddWeightInfo* iconf) noexcept(false);

  void fetchData(std::map<EcalLogicID, FEConfigOddWeightGroupDat>* fillMap,
                 FEConfigOddWeightInfo* iconf) noexcept(false);

  // User data
  int m_group_id;
  float m_w0;
  float m_w1;
  float m_w2;
  float m_w3;
  float m_w4;
  float m_w5;
};

#endif
