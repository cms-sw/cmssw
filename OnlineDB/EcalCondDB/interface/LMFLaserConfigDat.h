#ifndef LMFLASERCFGDAT_H
#define LMFLASERCFGDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LMFLaserConfigDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  LMFLaserConfigDat();
  ~LMFLaserConfigDat();

  // User data methods
  inline std::string getTable() { return "LMF_LASER_CONFIG_DAT"; }

  inline void setWavelength(int x) { m_wl = x; }
  inline int getWavelength() const { return m_wl; }
  inline void setVFEGain(int x) { m_vfe_gain = x; }
  inline int getVFEGain() const { return m_vfe_gain; }
  inline void setPNGain(int x) { m_pn_gain = x; }
  inline int getPNGain() const { return m_pn_gain; }

  inline void setAttenuator(float x) { m_attenuator = x; }
  inline float getAttenuator() const { return m_attenuator; }
  inline void setPower(float x) { m_power = x; }
  inline float getPower() const { return m_power; }
  inline void setCurrent(float x) { m_current = x; }
  inline float getCurrent() const { return m_current; }
  inline void setDelay1(float x) { m_delay1 = x; }
  inline float getDelay1() const { return m_delay1; }
  inline void setDelay2(float x) { m_delay2 = x; }
  inline float getDelay2() const { return m_delay2; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const LMFLaserConfigDat* item, LMFRunIOV* iov)
    throw(std::runtime_error);

void writeArrayDB(const std::map< EcalLogicID, LMFLaserConfigDat >* data, LMFRunIOV* iov)
  throw(std::runtime_error);


  void fetchData(std::map< EcalLogicID, LMFLaserConfigDat >* fillVec, LMFRunIOV* iov)
     throw(std::runtime_error);

  // User data
  int m_wl;
  int m_vfe_gain;
  int m_pn_gain;
  float m_power;
  float m_attenuator;
  float m_current;
  float m_delay1;
  float m_delay2;

};

#endif
