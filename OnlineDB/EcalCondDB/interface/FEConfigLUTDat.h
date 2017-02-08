#ifndef FECONFLUTDAT_H
#define FECONFLUTDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLUTInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigLUTDat : public IDataItem {
 public:
  friend class EcalCondDBInterface; // XXX temp should not need
  FEConfigLUTDat();
  ~FEConfigLUTDat();

  // User data methods
  inline std::string getTable() { return "FE_CONFIG_LUT_DAT"; }

  inline void setLUTGroupId(int x) { m_group_id = x; }
  inline int getLUTGroupId() const { return m_group_id; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const FEConfigLUTDat* item, FEConfigLUTInfo* iconf)
    throw(std::runtime_error);


  void writeArrayDB(const std::map< EcalLogicID, FEConfigLUTDat>* data, FEConfigLUTInfo* iconf)
  throw(std::runtime_error);


  void fetchData(std::map< EcalLogicID, FEConfigLUTDat >* fillMap, FEConfigLUTInfo* iconf)
     throw(std::runtime_error);

  // User data
  int m_group_id;

};

#endif
