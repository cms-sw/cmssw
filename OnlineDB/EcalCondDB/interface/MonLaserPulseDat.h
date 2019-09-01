#ifndef MONLASERPULSEDAT_H
#define MONLASERPULSEDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonLaserPulseDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  MonLaserPulseDat();
  ~MonLaserPulseDat() override;

  // User data methods
  inline std::string getTable() override { return "MON_LASER_PULSE_DAT"; }

  inline void setPulseHeightMean(float p) { m_pulseHeightMean = p; }
  inline float getPulseHeightMean() const { return m_pulseHeightMean; }

  inline void setPulseHeightRMS(float p) { m_pulseHeightRMS = p; }
  inline float getPulseHeightRMS() const { return m_pulseHeightRMS; }

  inline void setPulseWidthMean(float p) { m_pulseWidthMean = p; }
  inline float getPulseWidthMean() const { return m_pulseWidthMean; }

  inline void setPulseWidthRMS(float p) { m_pulseWidthRMS = p; }
  inline float getPulseWidthRMS() const { return m_pulseWidthRMS; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const MonLaserPulseDat* item, MonRunIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, MonLaserPulseDat>* data, MonRunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, MonLaserPulseDat>* fillMap, MonRunIOV* iov) noexcept(false);

  // User data
  float m_pulseHeightMean;
  float m_pulseHeightRMS;
  float m_pulseWidthMean;
  float m_pulseWidthRMS;
};

#endif
