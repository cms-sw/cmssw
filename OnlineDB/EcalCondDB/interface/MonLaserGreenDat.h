#ifndef MONLASERGREENDAT_H
#define MONLASERGREENDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonLaserGreenDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  MonLaserGreenDat();
  ~MonLaserGreenDat() override;

  // User data methods
  inline std::string getTable() override { return "MON_LASER_GREEN_DAT"; }

  inline void setAPDMean(float mean) { m_apdMean = mean; }
  inline float getAPDMean() const { return m_apdMean; }

  inline void setAPDRMS(float rms) { m_apdRMS = rms; }
  inline float getAPDRMS() const { return m_apdRMS; }

  inline void setAPDOverPNMean(float mean) { m_apdOverPNMean = mean; }
  inline float getAPDOverPNMean() const { return m_apdOverPNMean; }

  inline void setAPDOverPNRMS(float rms) { m_apdOverPNRMS = rms; }
  inline float getAPDOverPNRMS() const { return m_apdOverPNRMS; }

  inline void setTaskStatus(bool status) { m_taskStatus = status; }
  inline bool getTaskStatus() const { return m_taskStatus; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const MonLaserGreenDat* item, MonRunIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, MonLaserGreenDat>* data, MonRunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, MonLaserGreenDat>* fillMap, MonRunIOV* iov) noexcept(false);

  // User data
  float m_apdMean;
  float m_apdRMS;
  float m_apdOverPNMean;
  float m_apdOverPNRMS;
  bool m_taskStatus;
};

#endif
