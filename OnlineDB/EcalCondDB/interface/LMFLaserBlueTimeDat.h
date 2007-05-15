#ifndef LMFLASERBLUETIMEDAT_H
#define LMFLASERBLUETIMEDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LMFLaserBlueTimeDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  LMFLaserBlueTimeDat();
  ~LMFLaserBlueTimeDat();

  // User data methods
  inline std::string getTable() { return "LMF_LASER_BLUE_TIME_DAT"; }
  inline void setTiming(float mean) { m_timing = mean; }
  inline float getTiming() const { return m_timing; }

  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const LMFLaserBlueTimeDat* item, LMFRunIOV* iov)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, LMFLaserBlueTimeDat >* fillVec, LMFRunIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_timing;
  
};

#endif
