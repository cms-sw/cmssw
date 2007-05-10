#ifndef LMFMATACQBLUEDAT_H
#define LMFMATACQBLUEDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LMFMatacqBlueDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  LMFMatacqBlueDat();
  ~LMFMatacqBlueDat();

  // User data methods
  inline std::string getTable() { return "LMF_MATACQ_BLUE_DAT"; }
  inline void setAmplitude(float peak) { m_amplitude = peak; }
  inline float getAmplitude() const { return m_amplitude; }

  inline void setWidth(float width) { m_width = width; }
  inline float getWidth() const { return m_width; }

  inline void setTimeOffset(float x) { m_timeoffset = x; }
  inline float getTimeOffset() const { return m_timeoffset; }

  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const LMFMatacqBlueDat* item, LMFRunIOV* iov)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, LMFMatacqBlueDat >* fillVec, LMFRunIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_amplitude;
  float m_width;
  float m_timeoffset;
  
};

#endif
