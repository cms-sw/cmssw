#ifndef LMFCALIBPRIMDAT_H
#define LMFCALIBPRIMDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LMFCalibPrimDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  LMFCalibPrimDat();
  ~LMFCalibPrimDat();

  // User data methods
  inline std::string getTable() { return "LMF_CALIB_PRIM_DAT"; }

  inline void setMean(float mean) { m_Mean = mean; }
  inline float getMean() const { return m_Mean; }

  inline void setRMS(float RMS) { m_RMS = RMS; }
  inline float getRMS() const { return m_RMS; }

  inline void setPeak(float x) { m_Peak = x; }
  inline float getPeak() const { return m_Peak; }

  inline void setFlag(int x) { m_Flag = x; }
  inline int getFlag() const { return m_Flag; }
  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const LMFCalibPrimDat* item, LMFRunIOV* iov)
    throw(std::runtime_error);
  
  void writeArrayDB(const std::map< EcalLogicID, LMFCalibPrimDat >* data, LMFRunIOV* iov)
     throw(runtime_error);

  void fetchData(std::map< EcalLogicID, LMFCalibPrimDat >* fillVec, LMFRunIOV* iov)
     throw(std::runtime_error);

  // User data
  int m_Flag;
  float m_RMS;
  float m_Mean;
  float m_Peak;

};

#endif
