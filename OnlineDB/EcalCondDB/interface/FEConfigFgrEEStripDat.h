#ifndef FECONFFGREESTRIPDAT_H
#define FECONFFGREESTRIPDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigFgrInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigFgrEEStripDat : public IDataItem {
 public:
  friend class EcalCondDBInterface; // XXX temp should not need
  FEConfigFgrEEStripDat();
  ~FEConfigFgrEEStripDat();

  // User data methods
  inline std::string getTable() { return "FE_CONFIG_FGREEST_DAT"; }

  inline void setThreshold(int mean) { m_thresh = mean; }
  inline int getThreshold() const { return m_thresh; }
  inline void setLutFg(int mean) { m_lut_fg = mean; }
  inline int getLutFg() const { return m_lut_fg; }
  inline void setLUTFgr(int mean) { m_lut_fg = mean; }
  inline int getLUTFgr() const { return m_lut_fg; }
  inline void setLutFgr(int mean) { m_lut_fg = mean; }
  inline int getLutFgr() const { return m_lut_fg; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const FEConfigFgrEEStripDat* item, FEConfigFgrInfo* iconf)
    throw(std::runtime_error);


  void writeArrayDB(const std::map< EcalLogicID, FEConfigFgrEEStripDat>* data, FEConfigFgrInfo* iconf)
  throw(std::runtime_error);


  void fetchData(std::map< EcalLogicID, FEConfigFgrEEStripDat >* fillMap, FEConfigFgrInfo* iconf)
     throw(std::runtime_error);

  // User data
  int m_thresh;
  int m_lut_fg;

};

#endif
