#ifndef FECONFFGRGROUPDAT_H
#define FECONFFGRGROUPDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigFgrInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigFgrGroupDat : public IDataItem {
public:
  friend class EcalCondDBInterface;  // XXX temp should not need
  FEConfigFgrGroupDat();
  ~FEConfigFgrGroupDat() override;

  // User data methods
  inline std::string getTable() override { return "FE_CONFIG_FGR_PER_GROUP_DAT"; }

  inline void setFgrGroupId(int x) { m_group_id = x; }
  inline int getFgrGroupId() const { return m_group_id; }

  inline void setThreshLow(float x) { m_thresh_low = x; }
  inline float getThreshLow() const { return m_thresh_low; }
  inline void setThreshHigh(float x) { m_thresh_high = x; }
  inline float getThreshHigh() const { return m_thresh_high; }
  inline void setRatioLow(float x) { m_ratio_low = x; }
  inline float getRatioLow() const { return m_ratio_low; }
  inline void setRatioHigh(float x) { m_ratio_high = x; }
  inline float getRatioHigh() const { return m_ratio_high; }
  inline void setLUTValue(int x) { m_lut = x; }
  inline int getLUTValue() const { return m_lut; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const FEConfigFgrGroupDat* item, FEConfigFgrInfo* iconf) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, FEConfigFgrGroupDat>* data, FEConfigFgrInfo* iconf) noexcept(false);

  void fetchData(std::map<EcalLogicID, FEConfigFgrGroupDat>* fillMap, FEConfigFgrInfo* iconf) noexcept(false);

  // User data
  int m_group_id;
  float m_thresh_low;
  float m_thresh_high;
  float m_ratio_low;
  float m_ratio_high;
  int m_lut;
};

#endif
