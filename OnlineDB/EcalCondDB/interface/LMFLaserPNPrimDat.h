#ifndef LMFLASERPNPRIMDAT_H
#define LMFLASERPNPRIMDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LMFLaserPNPrimDat : public IDataItem {

 public:

// GHM
  enum laserColorNum { iBlue=0, iGreen, iRed, iIRed };

  friend class EcalCondDBInterface;
  LMFLaserPNPrimDat();
  ~LMFLaserPNPrimDat();

  // User data methods
  inline std::string getTable()  // GHM
  {
    switch( _color )
      {
      case iBlue:  return "LMF_LASER_BLUE_PN_PRIM_DAT"; 
      case iGreen: return "LMF_LASER_GREEN_PN_PRIM_DAT"; 
      case iRed:   return "LMF_LASER_RED_PN_PRIM_DAT"; 
      case iIRed:  return "LMF_LASER_IRED_PN_PRIM_DAT"; 
      }
  // default
    return "LMF_LASER_BLUE_PN_PRIM_DAT"; 
  }

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

  static void setColor( int color );  // GHM
  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const LMFLaserPNPrimDat* item, LMFRunIOV* iov)
    throw(std::runtime_error);
  
  void writeArrayDB(const std::map< EcalLogicID, LMFLaserPNPrimDat >* data, LMFRunIOV* iov)
     throw(runtime_error);

  void fetchData(std::map< EcalLogicID, LMFLaserPNPrimDat >* fillVec, LMFRunIOV* iov)
     throw(std::runtime_error);

  static int _color;  // GHM

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
