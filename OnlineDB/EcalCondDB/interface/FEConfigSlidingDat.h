#ifndef FECONFSLIDINGDAT_H
#define FECONFSLIDINGDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigSlidingInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigSlidingDat : public IDataItem {
 public:
  friend class EcalCondDBInterface; // XXX temp should not need
  FEConfigSlidingDat();
  ~FEConfigSlidingDat();

  // User data methods
  inline std::string getTable() { return "FE_CONFIG_SLIDING_DAT"; }

  inline void setSliding(float mean) { m_sliding = mean; }
  inline float getSliding() const { return m_sliding; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const FEConfigSlidingDat* item, FEConfigSlidingInfo* iconf)
    throw(std::runtime_error);


  void writeArrayDB(const std::map< EcalLogicID, FEConfigSlidingDat>* data, FEConfigSlidingInfo* iconf)
  throw(std::runtime_error);


  void fetchData(std::map< EcalLogicID, FEConfigSlidingDat >* fillMap, FEConfigSlidingInfo* iconf)
     throw(std::runtime_error);

  // User data
  float m_sliding;

};

#endif
