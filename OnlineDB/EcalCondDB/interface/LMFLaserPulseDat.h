#ifndef LMFLASERPULSEDAT_H
#define LMFLASERPULSEDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LMFLaserPulseDat : public IDataItem {

 public:

// GHM
  enum laserColorNum { iBlue=0, iGreen, iRed, iIRed };

  friend class EcalCondDBInterface;
  LMFLaserPulseDat();
  ~LMFLaserPulseDat();

  // User data methods
  inline std::string getTable()  // GHM
  {
    switch( _color )
      {
      case iBlue:  return "LMF_LASER_BLUE_PULSE_DAT"; 
      case iGreen: return "LMF_LASER_GREEN_PULSE_DAT"; 
      case iRed:   return "LMF_LASER_RED_PULSE_DAT"; 
      case iIRed:  return "LMF_LASER_IRED_PULSE_DAT"; 
      }
  // default
    return "LMF_LASER_BLUE_PULSE_DAT"; 
  }
  inline void setAmplitude(float x) { m_ampl = x; }
  inline float getAmplitude() const { return m_ampl; }
  inline void setTime(float x) { m_time = x; }
  inline float getTime() const { return m_time; }
  inline void setRise(float x) { m_rise = x; }
  inline float getRise() const { return m_rise; }
  inline void setFWHM(float x) { m_fwhm = x; }
  inline float getFWHM() const { return m_fwhm; }
  inline void setFW20(float x) { m_fw20 = x; }
  inline float getFW20() const { return m_fw20; }
  inline void setFW80(float x) { m_fw80 = x; }
  inline float getFW80() const { return m_fw80; }
  inline void setSliding(float x) { m_sliding = x; }
  inline float getSliding() const { return m_sliding; }

//  inline void setFitMethod(std::string x) { m_fit_method = x; }
//  inline std::string getFitMethod() const { return m_fit_method; }
  inline void setFitMethod(int x) { m_fit_method = x; }
  inline int  getFitMethod() const { return m_fit_method; }
  
  static void setColor( int color );  // GHM
  
 private:

  static int _color; // GHM

  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const LMFLaserPulseDat* item, LMFRunIOV* iov)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, LMFLaserPulseDat >* fillVec, LMFRunIOV* iov)
     throw(std::runtime_error);

  // User data
//  std::string m_fit_method;
  int m_fit_method;
  float m_ampl;
  float m_time;
  float m_rise;
  float m_fwhm;
  float m_fw20;
  float m_fw80;
  float m_sliding;
  
};

#endif
