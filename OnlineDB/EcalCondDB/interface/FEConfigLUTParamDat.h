#ifndef FECONFLUTPARAMDAT_H
#define FECONFLUTPARAMDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLUTInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigLUTParamDat : public IDataItem {
public:
  friend class EcalCondDBInterface;  // XXX temp should not need
  FEConfigLUTParamDat();
  ~FEConfigLUTParamDat() override;

  // User data methods
  inline std::string getTable() override { return "FE_CONFIG_LUTPARAM_DAT"; }

  inline void setETSat(float x) { m_etsat = x; }
  inline void setTTThreshlow(float x) { m_tthreshlow = x; }
  inline void setTTThreshhigh(float x) { m_tthreshhigh = x; }

  inline float getETSat() const { return m_etsat; }
  inline float getTTThreshlow() const { return m_tthreshlow; }
  inline float getTTThreshhigh() const { return m_tthreshhigh; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const FEConfigLUTParamDat* item, FEConfigLUTInfo* iconf) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, FEConfigLUTParamDat>* data, FEConfigLUTInfo* iconf) noexcept(false);

  void fetchData(std::map<EcalLogicID, FEConfigLUTParamDat>* fillMap, FEConfigLUTInfo* iconf) noexcept(false);

  // User data
  float m_etsat;
  float m_tthreshlow;
  float m_tthreshhigh;
};

#endif
