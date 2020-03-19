#ifndef MONLED2DAT_H
#define MONLED2DAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonLed2Dat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  MonLed2Dat();
  ~MonLed2Dat() override;

  // User data methods
  inline std::string getTable() override { return "MON_LED2_DAT"; }

  inline void setVPTMean(float mean) { m_vptMean = mean; }
  inline float getVPTMean() const { return m_vptMean; }

  inline void setVPTRMS(float rms) { m_vptRMS = rms; }
  inline float getVPTRMS() const { return m_vptRMS; }

  inline void setVPTOverPNMean(float mean) { m_vptOverPNMean = mean; }
  inline float getVPTOverPNMean() const { return m_vptOverPNMean; }

  inline void setVPTOverPNRMS(float rms) { m_vptOverPNRMS = rms; }
  inline float getVPTOverPNRMS() const { return m_vptOverPNRMS; }

  inline void setTaskStatus(bool status) { m_taskStatus = status; }
  inline bool getTaskStatus() const { return m_taskStatus; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const MonLed2Dat* item, MonRunIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, MonLed2Dat>* data, MonRunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, MonLed2Dat>* fillMap, MonRunIOV* iov) noexcept(false);

  // User data
  float m_vptMean;
  float m_vptRMS;
  float m_vptOverPNMean;
  float m_vptOverPNRMS;
  bool m_taskStatus;
};

#endif
