#ifndef FECONFLINPARAMDAT_H
#define FECONFLINPARAMDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLinInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigLinParamDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  FEConfigLinParamDat();
  ~FEConfigLinParamDat() override;

  // User data methods
  inline std::string getTable() override { return "FE_CONFIG_LINPARAM_DAT"; }

  inline void setETSat(float x) { m_etsat = x; }

  inline float getETSat() const { return m_etsat; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const FEConfigLinParamDat* item, FEConfigLinInfo* iconf) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, FEConfigLinParamDat>* data, FEConfigLinInfo* iconf) noexcept(false);

  void fetchData(std::map<EcalLogicID, FEConfigLinParamDat>* fillMap, FEConfigLinInfo* iconf) noexcept(false);

  // User data
  float m_etsat;
};

#endif
