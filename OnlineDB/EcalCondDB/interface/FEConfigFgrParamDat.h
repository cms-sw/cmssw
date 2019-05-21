#ifndef FECONFFGRPARAMDAT_H
#define FECONFFGRPARAMDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigFgrInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigFgrParamDat : public IDataItem {
public:
  friend class EcalCondDBInterface;  // XXX temp should not need
  FEConfigFgrParamDat();
  ~FEConfigFgrParamDat() override;

  // User data methods
  inline std::string getTable() override { return "FE_CONFIG_FGRPARAM_DAT"; }

  inline void setFGlowthresh(float x) { m_fglowthresh = x; }
  inline void setFGhighthresh(float x) { m_fghighthresh = x; }
  inline void setFGlowratio(float x) { m_lowratio = x; }
  inline void setFGhighratio(float x) { m_highratio = x; }

  inline float getFGlowthresh() const { return m_fglowthresh; }
  inline float getFGhighthresh() const { return m_fghighthresh; }
  inline float getFGlowratio() const { return m_lowratio; }
  inline float getFGhighratio() const { return m_highratio; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const FEConfigFgrParamDat* item, FEConfigFgrInfo* iconf) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, FEConfigFgrParamDat>* data, FEConfigFgrInfo* iconf) noexcept(false);

  void fetchData(std::map<EcalLogicID, FEConfigFgrParamDat>* fillMap, FEConfigFgrInfo* iconf) noexcept(false);

  // User data
  float m_fglowthresh;
  float m_fghighthresh;
  float m_lowratio;
  float m_highratio;
};

#endif
