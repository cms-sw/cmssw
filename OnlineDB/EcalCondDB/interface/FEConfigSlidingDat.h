#ifndef FECONFSLIDINGDAT_H
#define FECONFSLIDINGDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigSlidingInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigSlidingDat : public IDataItem {
public:
  friend class EcalCondDBInterface;  // XXX temp should not need
  FEConfigSlidingDat();
  ~FEConfigSlidingDat() override;

  // User data methods
  inline std::string getTable() override { return "FE_CONFIG_SLIDING_DAT"; }

  inline void setSliding(float mean) { m_sliding = mean; }
  inline float getSliding() const { return m_sliding; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const FEConfigSlidingDat* item, FEConfigSlidingInfo* iconf) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, FEConfigSlidingDat>* data, FEConfigSlidingInfo* iconf) noexcept(false);

  void fetchData(std::map<EcalLogicID, FEConfigSlidingDat>* fillMap, FEConfigSlidingInfo* iconf) noexcept(false);

  // User data
  float m_sliding;
};

#endif
