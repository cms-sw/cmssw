#ifndef FECONFFGREESTRIPDAT_H
#define FECONFFGREESTRIPDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigFgrInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigFgrEEStripDat : public IDataItem {
public:
  friend class EcalCondDBInterface;  // XXX temp should not need
  FEConfigFgrEEStripDat();
  ~FEConfigFgrEEStripDat() override;

  // User data methods
  inline std::string getTable() override { return "FE_CONFIG_FGREEST_DAT"; }

  inline void setThreshold(unsigned int mean) { m_thresh = mean; }
  inline unsigned int getThreshold() const { return m_thresh; }
  inline void setLutFg(unsigned int mean) { m_lut_fg = mean; }
  inline unsigned int getLutFg() const { return m_lut_fg; }
  inline void setLUTFgr(unsigned int mean) { m_lut_fg = mean; }
  inline unsigned int getLUTFgr() const { return m_lut_fg; }
  inline void setLutFgr(unsigned int mean) { m_lut_fg = mean; }
  inline unsigned int getLutFgr() const { return m_lut_fg; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const FEConfigFgrEEStripDat* item, FEConfigFgrInfo* iconf) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, FEConfigFgrEEStripDat>* data, FEConfigFgrInfo* iconf) noexcept(false);

  void fetchData(std::map<EcalLogicID, FEConfigFgrEEStripDat>* fillMap, FEConfigFgrInfo* iconf) noexcept(false);

  // User data
  unsigned int m_thresh;
  unsigned int m_lut_fg;
};

#endif
