#ifndef LMFLASERIREDRAWDAT_H
#define LMFLASERIREDRAWDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LMFLaserIRedRawDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  LMFLaserIRedRawDat();
  ~LMFLaserIRedRawDat();

  // User data methods
  inline std::string getTable() { return "LMF_LASER_IRED_RAW_DAT"; }

  inline void setAPDPeak(float peak) { m_apdPeak = peak; }
  inline float getAPDPeak() const { return m_apdPeak; }

  inline void setAPDErr(float err) { m_apdErr = err; }
  inline float getAPDErr() const { return m_apdErr; }

  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const LMFLaserIRedRawDat* item, LMFRunIOV* iov)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, LMFLaserIRedRawDat >* fillVec, LMFRunIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_apdPeak;
  float m_apdErr;
  
};

#endif
