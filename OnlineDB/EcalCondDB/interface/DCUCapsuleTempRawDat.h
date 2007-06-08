#ifndef DCUCAPSULETEMPRAWDAT_H
#define DCUCAPSULETEMPRAWDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/DCUTag.h"
#include "OnlineDB/EcalCondDB/interface/DCUIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class DCUCapsuleTempRawDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  DCUCapsuleTempRawDat();
  ~DCUCapsuleTempRawDat();

  // User data methods
  inline std::string getTable() { return "DCU_CAPSULE_TEMP_RAW_DAT"; }

  inline void setCapsuleTempADC(float adc) { m_capsuleTempADC = adc; }
  inline float getCapsuleTempADC() const { return m_capsuleTempADC; }

  inline void setCapsuleTempRMS(float rms) { m_capsuleTempRMS = rms; }
  inline float getCapsuleTempRMS() const { return m_capsuleTempRMS; }
   
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const DCUCapsuleTempRawDat* item, DCUIOV* iov)
    throw(std::runtime_error);

  void writeArrayDB(const std::map< EcalLogicID, DCUCapsuleTempRawDat>* data, DCUIOV* iov)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, DCUCapsuleTempRawDat >* fillVec, DCUIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_capsuleTempADC;
  float m_capsuleTempRMS;
  
};

#endif
