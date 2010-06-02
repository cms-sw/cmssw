#ifndef LMFLASERBLUECOEFFDAT_H
#define LMFLASERBLUECOEFFDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LMFLaserBlueCoeffDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  LMFLaserBlueCoeffDat();
  ~LMFLaserBlueCoeffDat();

  // User data methods
  inline std::string getTable() { return "LMF_LASER_BLUE_COEFF_DAT"; }

  inline void setXportCoeff(float c) { m_xportCoeff = c; }
  inline float getXportCoeff() const { return m_xportCoeff; }

  inline void setXportCoeffRMS(float rms) { m_xportCoeffRMS = rms; }
  inline float getXportCoeffRMS() const { return m_xportCoeffRMS; }

  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const LMFLaserBlueCoeffDat* item, LMFRunIOV* iov)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, LMFLaserBlueCoeffDat >* fillVec, LMFRunIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_xportCoeff;
  float m_xportCoeffRMS;
  
};

#endif
