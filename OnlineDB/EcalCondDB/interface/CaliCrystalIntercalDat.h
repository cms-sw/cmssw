#ifndef CALICRYSTALINTERCALDAT_H
#define CALICRYSTALINTERCALDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/CaliTag.h"
#include "OnlineDB/EcalCondDB/interface/CaliIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class CaliCrystalIntercalDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  CaliCrystalIntercalDat();
  ~CaliCrystalIntercalDat() override;

  // User data methods
  inline std::string getTable() override { return "CALI_CRYSTAL_INTERCAL_DAT"; }

  inline void setCali(float c) { m_cali = c; }
  inline float getCali() const { return m_cali; }

  inline void setCaliRMS(float c) { m_caliRMS = c; }
  inline float getCaliRMS() const { return m_caliRMS; }

  inline void setNumEvents(int n) { m_numEvents = n; }
  inline int getNumEvents() const { return m_numEvents; }

  inline void setTaskStatus(bool s) { m_taskStatus = s; }
  inline bool getTaskStatus() const { return m_taskStatus; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const CaliCrystalIntercalDat* item, CaliIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, CaliCrystalIntercalDat>* fillVec, CaliIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, CaliCrystalIntercalDat>* data, CaliIOV* iov) noexcept(false);

  // User data
  float m_cali;
  float m_caliRMS;
  int m_numEvents;
  bool m_taskStatus;
};

#endif
