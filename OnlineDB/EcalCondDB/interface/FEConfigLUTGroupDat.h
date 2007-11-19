#ifndef FECONFLUTGROUPDAT_H
#define FECONFLUTGROUPDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLUTInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigLUTGroupDat : public IDataItem {
 public:
  friend class EcalCondDBInterface; // XXX temp should not need
  FEConfigLUTGroupDat();
  ~FEConfigLUTGroupDat();

  // User data methods
  inline std::string getTable() { return "FE_CONFIG_LUT_PER_GROUP_DAT"; }

  inline void setLUTGroupId(int x) { m_group_id = x; }
  inline int getLUTGroupId() const { return m_group_id; }

  inline void setLUTValue(int i, int x) { m_lut[i] = x; }
  inline int getLUTValue(int i) const { return m_lut[i]; }



 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const FEConfigLUTGroupDat* item, FEConfigLUTInfo* iconf)
    throw(std::runtime_error);


  void writeArrayDB(const std::map< EcalLogicID, FEConfigLUTGroupDat>* data, FEConfigLUTInfo* iconf)
  throw(std::runtime_error);


  void fetchData(std::map< EcalLogicID, FEConfigLUTGroupDat >* fillMap, FEConfigLUTInfo* iconf)
     throw(std::runtime_error);

  // User data
  int m_group_id;
  int m_lut[1024];

};

#endif
