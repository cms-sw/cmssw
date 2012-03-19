#ifndef LMFLASERCONFIGDAT_H
#define LMFLASERCONFIGDAT_H

/*
 Last updated by  Giovanni.Organtini@roma1.infn.it 2010
 */

#include "OnlineDB/EcalCondDB/interface/LMFDat.h"

/**
 *   LMF_LASER_CONFIG_DAT interface
 */
class LMFLaserConfigDat : public LMFDat {
 public:
  LMFLaserConfigDat() : LMFDat() {
    m_tableName = "LMF_LASER_CONFIG_DAT";
    m_className = "LMFLaserConfigDat";
    m_keys["WAVELENGTH"] = 0;
    m_keys["VFE_GAIN"] = 1;
    m_keys["PN_GAIN"] = 2;
    m_keys["LSR_POWER"] = 3;
    m_keys["LSR_ATTENUATOR"] = 4;
    m_keys["LSR_CURRENT"] = 5;
    m_keys["LSR_DELAY_1"] = 6;
    m_keys["LSR_DELAY_2"] = 7;
  }
  LMFLaserConfigDat(EcalDBConnection *c) : LMFDat(c) {
    m_tableName = "LMF_LASER_CONFIG_DAT";
    m_className = "LMFLaserConfigDat";
    m_keys["WAVELENGTH"] = 0;
    m_keys["VFE_GAIN"] = 1;
    m_keys["PN_GAIN"] = 2;
    m_keys["LSR_POWER"] = 3;
    m_keys["LSR_ATTENUATOR"] = 4;
    m_keys["LSR_CURRENT"] = 5;
    m_keys["LSR_DELAY_1"] = 6;
    m_keys["LSR_DELAY_2"] = 7;
  }
  ~LMFLaserConfigDat() {}

  LMFLaserConfigDat& setWavelength(EcalLogicID &id, int w) {
    LMFDat::setData(id, "WAVELENGTH", w);
    return *this;
  }
  LMFLaserConfigDat& setVFEGain(EcalLogicID &id, float g) {
    LMFDat::setData(id, "VFE_GAIN", g);
    return *this;
  }
  LMFLaserConfigDat& setPNGain(EcalLogicID &id, float g) {
    LMFDat::setData(id, "PN_GAIN", g);
    return *this;
  }
  LMFLaserConfigDat& setLSRPower(EcalLogicID &id, float g) {
    LMFDat::setData(id, "LSR_POWER", g);
    return *this;
  }
  LMFLaserConfigDat& setLSRAttenuator(EcalLogicID &id, float g) {
    LMFDat::setData(id, "LSR_ATTENUATOR", g);
    return *this;
  }
  LMFLaserConfigDat& setLSRCurrent(EcalLogicID &id, float g) {
    LMFDat::setData(id, "LSR_CURRENT", g);
    return *this;
  }
  LMFLaserConfigDat& setLSRDelay1(EcalLogicID &id, float g) {
    LMFDat::setData(id, "LSR_DELAY_1", g);
    return *this;
  }
  LMFLaserConfigDat& setLSRDelay2(EcalLogicID &id, float g) {
    LMFDat::setData(id, "LSR_DELAY_2", g);
    return *this;
  }
  LMFLaserConfigDat& setData(EcalLogicID &id, float w, float g, float pnga, 
			     float p, float a, float c, float d1, float d2) {
    LMFDat::setData(id, "WAVELENGTH", w);
    LMFDat::setData(id, "VFE_GAIN", g);
    LMFDat::setData(id, "PN_GAIN", pnga);
    LMFDat::setData(id, "LSR_POWER", p);
    LMFDat::setData(id, "LSR_ATTENUATOR", a);
    LMFDat::setData(id, "LSR_CURRENT", c);
    LMFDat::setData(id, "LSR_DELAY_1", d1);
    LMFDat::setData(id, "LSR_DELAY_2", d2);
    return *this;
  }
  LMFLaserConfigDat& setData(EcalLogicID &id, std::vector<float> v) {
    LMFDat::setData(id, v);
    return *this;
  }

  float getWavelength(EcalLogicID &id) {
    return getData(id, "WAVELENGTH");
  }
  float getVFEGain(EcalLogicID &id) {
    return getData(id, "VFE_GAIN");
  }
  float getPNGain(EcalLogicID &id) {
    return getData(id, "PN_GAIN");
  }
  float getLSRPower(EcalLogicID &id) {
    return getData(id, "LSR_POWER");
  }
  float getLSRAttenuator(EcalLogicID &id) {
    return getData(id, "LSR_ATTENUATOR");
  }
  float getLSRCURRENT(EcalLogicID &id) {
    return getData(id, "LSR_CURRENT");
  }
  float getLSRDelay1(EcalLogicID &id) {
    return getData(id, "LSR_DELAY_1");
  }
  float getLSRDelay2(EcalLogicID &id) {
    return getData(id, "LSR_DELAY_2");
  }

  
 private:

};

#endif
