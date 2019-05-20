#ifndef MONDELAYSTTDAT_H
#define MONDELAYSTTDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonDelaysTTDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  MonDelaysTTDat();
  ~MonDelaysTTDat() override;

  // User data methods
  inline std::string getTable() override { return "MON_DELAYS_TT_DAT"; }

  inline void setDelayMean(float mean) { m_delayMean = mean; }
  inline float getDelayMean() const { return m_delayMean; }

  inline void setDelayRMS(float rms) { m_delayRMS = rms; }
  inline float getDelayRMS() const { return m_delayRMS; }

  inline void setTaskStatus(bool status) { m_taskStatus = status; }
  inline bool getTaskStatus() const { return m_taskStatus; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const MonDelaysTTDat* item, MonRunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, MonDelaysTTDat>* fillVec, MonRunIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, MonDelaysTTDat>* data, MonRunIOV* iov) noexcept(false);

  // User data
  float m_delayMean;
  float m_delayRMS;
  bool m_taskStatus;
};

#endif
