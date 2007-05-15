#ifndef LMFLASERBLUERAWDAT_H
#define LMFLASERBLUERAWDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LMFLaserBlueRawDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  LMFLaserBlueRawDat();
  ~LMFLaserBlueRawDat();

  // User data methods
  inline std::string getTable() { return "LMF_LASER_BLUE_RAW_DAT"; }

  inline void setAPDPeak(float peak) { m_apdPeak = peak; }
  inline float getAPDPeak() const { return m_apdPeak; }

  inline void setAPDErr(float err) { m_apdErr = err; }
  inline float getAPDErr() const { return m_apdErr; }

  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const LMFLaserBlueRawDat* item, LMFRunIOV* iov)
    throw(std::runtime_error);

 void writeArrayDB(const std::map< EcalLogicID, LMFLaserBlueRawDat >* data, LMFRunIOV* iov)
   throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, LMFLaserBlueRawDat >* fillVec, LMFRunIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_apdPeak;
  float m_apdErr;
  
};

#endif
