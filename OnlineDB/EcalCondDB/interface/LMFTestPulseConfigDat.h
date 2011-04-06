#ifndef LMFTESTPULSECONFIGDAT_H
#define LMFTESTPULSECONFIGDAT_H

/*
 Last updated by  Giovanni.Organtini@roma1.infn.it 2010
 */

#include "OnlineDB/EcalCondDB/interface/LMFDat.h"

/**
 *   LMF_TEST_PULSE_CONFIG_DAT interface
 */
class LMFTestPulseConfigDat : public LMFDat {
 public:
  LMFTestPulseConfigDat() : LMFDat() {
    m_tableName = "LMF_TEST_PULSE_CONFIG_DAT";
    m_className = "LMFTestPulseConfigDat";
    m_keys["VFE_GAIN"] = 0;
    m_keys["DAC_MGPA"] = 1;
    m_keys["PN_GAIN"] = 2;
    m_keys["PN_VINJ"] = 3;
  }
  LMFTestPulseConfigDat(EcalDBConnection *c) : LMFDat(c) {
    m_tableName = "LMF_TEST_PULSE_CONFIG_DAT";
    m_className = "LMFTestPulseConfigDat";
    m_keys["VFE_GAIN"] = 0;
    m_keys["DAC_MGPA"] = 1;
    m_keys["PN_GAIN"] = 2;
    m_keys["PN_VINJ"] = 3;
  }
  ~LMFTestPulseConfigDat() {}

  LMFTestPulseConfigDat& setVFEGain(EcalLogicID &id, float g) {
    LMFDat::setData(id, "VFE_GAIN", g);
    return *this;
  }
  LMFTestPulseConfigDat& setPNGain(EcalLogicID &id, float g) {
    LMFDat::setData(id, "PN_GAIN", g);
    return *this;
  }
  LMFTestPulseConfigDat& setDACMGPA(EcalLogicID &id, float g) {
    LMFDat::setData(id, "DAC_MGPA", g);
    return *this;
  }
  LMFTestPulseConfigDat& setPNVinj(EcalLogicID &id, float g) {
    LMFDat::setData(id, "PN_VINJ", g);
    return *this;
  }
  LMFTestPulseConfigDat& setData(EcalLogicID &id, float g, float d, float pnga, 
				 float pnv) {
    LMFDat::setData(id, "VFE_GAIN", g);
    LMFDat::setData(id, "DAC_MGPA", d);
    LMFDat::setData(id, "PN_GAIN", pnga);
    LMFDat::setData(id, "PN_VINJ", pnv);
    return *this;
  }
  LMFTestPulseConfigDat& setData(EcalLogicID &id, std::vector<float> v) {
    LMFDat::setData(id, v);
    return *this;
  }

  float getVFEGain(EcalLogicID &id) {
    return getData(id, "VFE_GAIN");
  }
  float getPNGain(EcalLogicID &id) {
    return getData(id, "PN_GAIN");
  }
  float getDACMGPA(EcalLogicID &id) {
    return getData(id, "DAC_MGPA");
  }
  float getPNVinj(EcalLogicID &id) {
    return getData(id, "PN_VINJ");
  }
  
 private:

};

#endif
