#ifndef RUNDCSMAGNET_H
#define RUNDCSMAGNET_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"
#include "OnlineDB/EcalCondDB/interface/DataReducer.h"
#include "OnlineDB/Oracle/interface/Oracle.h"

class RunDCSMagnetDat : public IDataItem {
public:
  typedef oracle::occi::ResultSet ResultSet;

  friend class EcalCondDBInterface;
  RunDCSMagnetDat();
  ~RunDCSMagnetDat() override;

  // User data methods
  inline std::string getTable() override { return "CMSFWMAGNET_LV"; }
  inline std::string getMagnetAccount() { return "CMS_DCS_ENV_PVSS_COND"; }
  inline void setMagnetCurrent(float t) { m_current = t; }
  inline float getMagnetCurrent() const { return m_current; }

  void setTime(const Tm& start);
  Tm getTime() const;

private:
  ResultSet* getMagnetRset();

  int nowMicroseconds();

  void fillTheMap(ResultSet*, std::map<EcalLogicID, RunDCSMagnetDat>*);

  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const RunDCSMagnetDat* item, RunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, RunDCSMagnetDat>* fillMap, RunIOV* iov) noexcept(false);

  void fetchLastData(std::map<EcalLogicID, RunDCSMagnetDat>* fillMap) noexcept(false);

  // User data
  float m_current;
  Tm m_time;
};

#endif
