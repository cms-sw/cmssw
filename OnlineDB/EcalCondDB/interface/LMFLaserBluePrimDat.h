#ifndef LMFLASERBLUEPRIMDAT_H
#define LMFLASERBLUEPRIMDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LMFLaserBluePrimDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  LMFLaserBluePrimDat();
  ~LMFLaserBluePrimDat();

  // User data methods
  inline std::string getTable() { return "LMF_LASER_BLUE_PRIM_DAT"; }

  inline void setMean(float mean) { m_Mean = mean; }
  inline float getMean() const { return m_Mean; }

  inline void setRMS(float RMS) { m_RMS = RMS; }
  inline float getRMS() const { return m_RMS; }

  inline void setPeak(float x) { m_Peak = x; }
  inline float getPeak() const { return m_Peak; }

  inline void setFlag(int x) { m_Flag = x; }
  inline int getFlag() const { return m_Flag; }
  //
  inline void setAPDOverPNAMean(float mean) { m_apdOverPNAMean = mean; }
  inline float getAPDOverPNAMean() const { return m_apdOverPNAMean; }

  inline void setAPDOverPNARMS(float RMS) { m_apdOverPNARMS = RMS; }
  inline float getAPDOverPNARMS() const { return m_apdOverPNARMS; }

  inline void setAPDOverPNAPeak(float x) { m_apdOverPNAPeak = x; }
  inline float getAPDOverPNAPeak() const { return m_apdOverPNAPeak; }

  inline void setAPDOverPNBMean(float mean) { m_apdOverPNBMean = mean; }
  inline float getAPDOverPNBMean() const { return m_apdOverPNBMean; }

  inline void setAPDOverPNBRMS(float RMS) { m_apdOverPNBRMS = RMS; }
  inline float getAPDOverPNBRMS() const { return m_apdOverPNBRMS; }

  inline void setAPDOverPNBPeak(float x) { m_apdOverPNBPeak = x; }
  inline float getAPDOverPNBPeak() const { return m_apdOverPNBPeak; }

  inline void setAPDOverPNMean(float mean) { m_apdOverPNMean = mean; }
  inline float getAPDOverPNMean() const { return m_apdOverPNMean; }

  inline void setAPDOverPNRMS(float RMS) { m_apdOverPNRMS = RMS; }
  inline float getAPDOverPNRMS() const { return m_apdOverPNRMS; }

  inline void setAPDOverPNPeak(float x) { m_apdOverPNPeak = x; }
  inline float getAPDOverPNPeak() const { return m_apdOverPNPeak; }
  
  inline void setAlpha(float mean) { m_Alpha = mean; }
  inline float getAlpha() const { return m_Alpha; }

  inline void setBeta(float x) { m_Beta = x; }
  inline float getBeta() const { return m_Beta; }

  inline void setShapeCor(float x) { m_ShapeCor = x; }
  inline float getShapeCor() const { return m_ShapeCor; }



 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const LMFLaserBluePrimDat* item, LMFRunIOV* iov)
    throw(std::runtime_error);

void writeArrayDB(const std::map< EcalLogicID, LMFLaserBluePrimDat >* data, LMFRunIOV* iov)
  throw(std::runtime_error);


  void fetchData(std::map< EcalLogicID, LMFLaserBluePrimDat >* fillVec, LMFRunIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_Mean;
  float m_RMS;
  float m_Peak;
  int m_Flag;
  float m_apdOverPNAMean;
  float m_apdOverPNARMS;
  float m_apdOverPNAPeak;
  float m_apdOverPNBMean;
  float m_apdOverPNBRMS;
  float m_apdOverPNBPeak;
  float m_apdOverPNMean;
  float m_apdOverPNRMS;
  float m_apdOverPNPeak;
  float m_Alpha;
  float m_Beta;
  float m_ShapeCor;
  
};

#endif
