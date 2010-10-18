#ifndef LMFPNBLUEPRIMDAT_H
#define LMFPNBLUEPRIMDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LMFPNBluePrimDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  LMFPNBluePrimDat();
  ~LMFPNBluePrimDat();

  // User data methods
  inline std::string getTable() { return "LMF_Laser_blue_PN_PRIM_DAT"; }

  inline void setMean(float mean) { m_Mean = mean; }
  inline float getMean() const { return m_Mean; }

  inline void setRMS(float RMS) { m_RMS = RMS; }
  inline float getRMS() const { return m_RMS; }

  inline void setPeak(float x) { m_Peak = x; }
  inline float getPeak() const { return m_Peak; }

  inline void setFlag(int x) { m_Flag = x; }
  inline int getFlag() const { return m_Flag; }
  //
  inline void setPNAOverPNBMean(float mean) { m_PNAOverPNBMean = mean; }
  inline float getPNAOverPNBMean() const { return m_PNAOverPNBMean; }
  inline void setPNAOverPNBRMS(float mean) { m_PNAOverPNBRMS = mean; }
  inline float getPNAOverPNBRMS() const { return m_PNAOverPNBRMS; }
  inline void setPNAOverPNBPeak(float mean) { m_PNAOverPNBPeak = mean; }
  inline float getPNAOverPNBPeak() const { return m_PNAOverPNBPeak; }

  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const LMFPNBluePrimDat* item, LMFRunIOV* iov)
    throw(std::runtime_error);
  
  void writeArrayDB(const std::map< EcalLogicID, LMFPNBluePrimDat >* data, LMFRunIOV* iov)
     throw(runtime_error);

  void fetchData(std::map< EcalLogicID, LMFPNBluePrimDat >* fillVec, LMFRunIOV* iov)
     throw(std::runtime_error);

  // User data
  int m_Flag;
  float m_RMS;
  float m_Mean;
  float m_Peak;
  float m_PNAOverPNBMean;
  float m_PNAOverPNBRMS;
  float m_PNAOverPNBPeak;

};

#endif
