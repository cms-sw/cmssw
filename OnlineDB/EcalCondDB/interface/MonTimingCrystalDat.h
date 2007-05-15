#ifndef MONTIMINGCRYSTALDAT_H
#define MONTIMINGCRYSTALDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonTimingCrystalDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  MonTimingCrystalDat();
  ~MonTimingCrystalDat();

  // User data methods
  inline std::string getTable() { return "MON_TIMING_CRYSTAL_DAT"; }

  inline void setTimingMean(float mean) { m_timingMean = mean; }
  inline float getTimingMean() const { return m_timingMean; }
  
  inline void setTimingRMS(float rms) { m_timingRMS = rms; }
  inline float getTimingRMS() const { return m_timingRMS; }
  

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const MonTimingCrystalDat* item, MonRunIOV* iov)
    throw(std::runtime_error);

  void writeArrayDB(const std::map< EcalLogicID, MonTimingCrystalDat >* data, MonRunIOV* iov)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, MonTimingCrystalDat >* fillVec, MonRunIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_timingMean;
  float m_timingRMS;
  
};

#endif
