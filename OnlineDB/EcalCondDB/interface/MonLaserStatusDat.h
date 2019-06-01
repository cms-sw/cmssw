#ifndef MONLASERSTATUSDAT_H
#define MONLASERSTATUSDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonLaserStatusDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  MonLaserStatusDat();
  ~MonLaserStatusDat() override;

  // User data methods
  inline std::string getTable() override { return "MON_LASER_STATUS_DAT"; }
  inline void setLaserPower(float p) { m_laserPower = p; }
  inline float getLaserPower() const { return m_laserPower; }

  inline void setLaserFilter(float p) { m_laserFilter = p; }
  inline float getLaserFilter() const { return m_laserFilter; }

  inline void setLaserWavelength(float p) { m_laserWavelength = p; }
  inline float getLaserWavelength() const { return m_laserWavelength; }

  inline void setLaserFanout(float p) { m_laserFanout = p; }
  inline float getLaserFanout() const { return m_laserFanout; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const MonLaserStatusDat* item, MonRunIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, MonLaserStatusDat>* data, MonRunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, MonLaserStatusDat>* fillMap, MonRunIOV* iov) noexcept(false);

  // User data
  float m_laserPower;
  float m_laserFilter;
  float m_laserWavelength;
  float m_laserFanout;
};

#endif
