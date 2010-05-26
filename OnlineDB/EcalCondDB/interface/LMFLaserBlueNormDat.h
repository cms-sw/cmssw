#ifndef LMFLASERBLUENORMDAT_H
#define LMFLASERBLUENORMDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LMFLaserBlueNormDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  LMFLaserBlueNormDat();
  ~LMFLaserBlueNormDat();

  // User data methods
  inline std::string getTable() { return "LMF_LASER_BLUE_NORM_DAT"; }

  inline void setAPDOverPNAMean(float mean) { m_apdOverPNAMean = mean; }
  inline float getAPDOverPNAMean() const { return m_apdOverPNAMean; }

  inline void setAPDOverPNARMS(float RMS) { m_apdOverPNARMS = RMS; }
  inline float getAPDOverPNARMS() const { return m_apdOverPNARMS; }

  inline void setAPDOverPNBMean(float mean) { m_apdOverPNBMean = mean; }
  inline float getAPDOverPNBMean() const { return m_apdOverPNBMean; }

  inline void setAPDOverPNBRMS(float RMS) { m_apdOverPNBRMS = RMS; }
  inline float getAPDOverPNBRMS() const { return m_apdOverPNBRMS; }

  inline void setAPDOverPNMean(float mean) { m_apdOverPNMean = mean; }
  inline float getAPDOverPNMean() const { return m_apdOverPNMean; }

  inline void setAPDOverPNRMS(float RMS) { m_apdOverPNRMS = RMS; }
  inline float getAPDOverPNRMS() const { return m_apdOverPNRMS; }

  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const LMFLaserBlueNormDat* item, LMFRunIOV* iov)
    throw(std::runtime_error);

void writeArrayDB(const std::map< EcalLogicID, LMFLaserBlueNormDat >* data, LMFRunIOV* iov)
  throw(std::runtime_error);


  void fetchData(std::map< EcalLogicID, LMFLaserBlueNormDat >* fillVec, LMFRunIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_apdOverPNAMean;
  float m_apdOverPNARMS;
  float m_apdOverPNBMean;
  float m_apdOverPNBRMS;
  float m_apdOverPNMean;
  float m_apdOverPNRMS;
  
};

#endif
