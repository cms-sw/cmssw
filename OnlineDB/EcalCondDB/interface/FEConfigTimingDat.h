#ifndef FECONFTimingDAT_H
#define FECONFTimingDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigTimingInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigTimingDat : public IDataItem {
 public:
  friend class EcalCondDBInterface; // XXX temp should not need
  FEConfigTimingDat();
  ~FEConfigTimingDat();

  // User data methods
  inline std::string getTable() { return "FE_CONFIG_Time_DAT"; }

  inline void setTimingPar1(int x) { m_par1 = x; }
  inline int getTimingPar1() const { return m_par1; }
  inline void setTimingPar2(int x) { m_par2 = x; }
  inline int getTimingPar2() const { return m_par2; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const FEConfigTimingDat* item, FEConfigTimingInfo* iconf)
    throw(std::runtime_error);

  void writeArrayDB(const std::map< EcalLogicID, FEConfigTimingDat>* data, FEConfigTimingInfo* iconf)
  throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, FEConfigTimingDat >* fillMap, FEConfigTimingInfo* iconf)
     throw(std::runtime_error);

  // User data
  int m_par1;
  int m_par2;

};

#endif
