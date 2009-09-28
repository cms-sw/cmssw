#ifndef FECONFFGREETOWERDAT_H
#define FECONFFGREETOWERDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigFgrInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigFgrEETowerDat : public IDataItem {
 public:
  friend class EcalCondDBInterface; // XXX temp should not need
  FEConfigFgrEETowerDat();
  ~FEConfigFgrEETowerDat();

  // User data methods
  inline std::string getTable() { return "FE_CONFIG_FGREETT_DAT"; }

  inline void setLUTValue(int mean) { m_lut = mean; }
  inline int getLUTValue() const { return m_lut; }
  inline void setLutValue(int mean) { m_lut = mean; }
  inline int getLutValue() const { return m_lut; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const FEConfigFgrEETowerDat* item, FEConfigFgrInfo* iconf)
    throw(std::runtime_error);


  void writeArrayDB(const std::map< EcalLogicID, FEConfigFgrEETowerDat>* data, FEConfigFgrInfo* iconf)
  throw(std::runtime_error);


  void fetchData(std::map< EcalLogicID, FEConfigFgrEETowerDat >* fillMap, FEConfigFgrInfo* iconf)
     throw(std::runtime_error);

  // User data

  int m_lut;

};

#endif
