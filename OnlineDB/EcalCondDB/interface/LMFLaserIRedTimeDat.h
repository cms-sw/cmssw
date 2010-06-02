#ifndef LMFLASERIREDTIMEDAT_H
#define LMFLASERIREDTIMEDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LMFLaserIRedTimeDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  LMFLaserIRedTimeDat();
  ~LMFLaserIRedTimeDat();

  // User data methods
 inline std::string getTable() { return "LMF_LASER_IRED_TIME_DAT"; }
  inline void setTiming(float mean) { m_timing = mean; }
  inline float getTiming() const { return m_timing; }

  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const LMFLaserIRedTimeDat* item, LMFRunIOV* iov)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, LMFLaserIRedTimeDat >* fillVec, LMFRunIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_timing;
  
};

#endif
