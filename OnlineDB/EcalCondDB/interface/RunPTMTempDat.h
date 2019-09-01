#ifndef RUNPTMTEMPDAT_H
#define RUNPTMTEMPDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class RunPTMTempDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  RunPTMTempDat();
  ~RunPTMTempDat() override;

  // User data methods
  inline std::string getTable() override { return "RUN_PTM_TEMP_DAT"; }
  inline void setTemperature(float t) { m_temperature = t; }
  inline float getTemperature() const { return m_temperature; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const RunPTMTempDat* item, RunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, RunPTMTempDat>* fillMap, RunIOV* iov) noexcept(false);

  // User data
  float m_temperature;
};

#endif
