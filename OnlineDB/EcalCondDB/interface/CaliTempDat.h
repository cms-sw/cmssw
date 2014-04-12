#ifndef CALITEMPDAT_H
#define CALITEMPDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/CaliTag.h"
#include "OnlineDB/EcalCondDB/interface/CaliIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class CaliTempDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  CaliTempDat();
  ~CaliTempDat();
  
  // User data methods
  inline std::string getTable() { return "CALI_TEMP_DAT"; }

  inline void setBeta(float c) { m_beta = c; }
  inline float getBeta() const { return m_beta; }

  inline void setR25(float c) { m_r25 = c; }
  inline float getR25() const { return m_r25; }

  inline void setOffset(float c) { m_offset = c; }
  inline float getOffset() const { return m_offset; }

  inline void setTaskStatus(bool s) { m_taskStatus = s; }
  inline bool getTaskStatus() const { return m_taskStatus; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);
  
  void writeDB(const EcalLogicID* ecid, const CaliTempDat* item, CaliIOV* iov)
    throw(std::runtime_error);
  
  void fetchData(std::map< EcalLogicID, CaliTempDat >* fillVec, CaliIOV* iov)
    throw(std::runtime_error);

  void writeArrayDB(const std::map< EcalLogicID, CaliTempDat >* data, CaliIOV* iov)
    throw(std::runtime_error);


  
  // User data
  float m_beta;
  float m_r25;
  float m_offset;
  bool m_taskStatus;
  
  
};

#endif
