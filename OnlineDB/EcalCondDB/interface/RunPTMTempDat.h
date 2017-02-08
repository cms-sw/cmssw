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
  ~RunPTMTempDat();

  // User data methods
  inline std::string getTable() { return "RUN_PTM_TEMP_DAT"; }
  inline void setTemperature(float t) { m_temperature = t; }
  inline float getTemperature() const { return m_temperature; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const RunPTMTempDat* item, RunIOV* iov )
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, RunPTMTempDat >* fillMap, RunIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_temperature;
};

#endif
