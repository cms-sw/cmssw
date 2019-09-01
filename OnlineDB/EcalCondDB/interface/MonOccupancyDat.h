#ifndef MONOCCUPANCYDAT_H
#define MONOCCUPANCYDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonOccupancyDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  MonOccupancyDat();
  ~MonOccupancyDat() override;

  // User data methods
  inline std::string getTable() override { return "MON_OCCUPANCY_DAT"; }

  void setEventsOverLowThreshold(int events) { m_eventsOverLowThreshold = events; }
  int getEventsOverLowThreshold() const { return m_eventsOverLowThreshold; }

  void setEventsOverHighThreshold(int events) { m_eventsOverHighThreshold = events; }
  int getEventsOverHighThreshold() const { return m_eventsOverHighThreshold; }

  void setAvgEnergy(float energy) { m_avgEnergy = energy; }
  float getAvgEnergy() const { return m_avgEnergy; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const MonOccupancyDat* item, MonRunIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, MonOccupancyDat>* data, MonRunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, MonOccupancyDat>* fillVec, MonRunIOV* iov) noexcept(false);

  // User data
  int m_eventsOverLowThreshold;
  int m_eventsOverHighThreshold;
  float m_avgEnergy;
};

#endif
