#ifndef LMFMATACQGREENDAT_H
#define LMFMATACQGREENDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LMFMatacqGreenDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  LMFMatacqGreenDat();
  ~LMFMatacqGreenDat();

  // User data methods
  inline std::string getTable() { return "LMF_MATACQ_GREEN_DAT"; }
  inline void setAmplitude(float peak) { m_amplitude = peak; }
  inline float getAmplitude() const { return m_amplitude; }

  inline void setWidth(float width) { m_width = width; }
  inline float getWidth() const { return m_width; }

  inline void setTimeOffset(float x) { m_timeoffset = x; }
  inline float getTimeOffset() const { return m_timeoffset; }

  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const LMFMatacqGreenDat* item, LMFRunIOV* iov)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, LMFMatacqGreenDat >* fillVec, LMFRunIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_amplitude;
  float m_width;
  float m_timeoffset;
  
};

#endif
