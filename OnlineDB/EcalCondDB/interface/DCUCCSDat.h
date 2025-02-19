#ifndef DCUCCSDAT_H
#define DCUCCSDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/DCUTag.h"
#include "OnlineDB/EcalCondDB/interface/DCUIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class DCUCCSDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  DCUCCSDat();
  ~DCUCCSDat();

  // User data methods
  inline std::string getTable() { return "DCU_CCS_DAT"; }

  inline void setM1VDD1(float temp) { m_m1_vdd1 = temp; }
  inline void setM2VDD1(float temp) { m_m2_vdd1 = temp; }
  inline void setM1VDD2(float temp) { m_m1_vdd2 = temp; }
  inline void setM2VDD2(float temp) { m_m2_vdd2 = temp; }
  inline void setVDD(float m1vdd1, float m1vdd2, float m2vdd1, float m2vdd2) {
    setM1VDD1(m1vdd1);
    setM1VDD2(m1vdd2);
    setM2VDD1(m2vdd1);
    setM2VDD2(m2vdd2);
  }
  inline void setM1Vinj(float temp) { m_m1_vinj = temp; }
  inline void setM2Vinj(float temp) { m_m2_vinj = temp; }
  inline void setVinj(float v1, float v2) {
    setM1Vinj(v1);
    setM2Vinj(v2);
  }
  inline void setM1Vcc(float temp) { m_m1_vcc = temp; }
  inline void setM2Vcc(float temp) { m_m2_vcc = temp; }
  inline void setVcc(float v1, float v2) {
    setM1Vcc(v1);
    setM2Vcc(v2);
  }
  inline void setM1DCUTemp(float temp) { m_m1_dcutemp = temp; }
  inline void setM2DCUTemp(float temp) { m_m2_dcutemp = temp; }
  inline void setDCUTemp(float t1, float t2) {
    setM1DCUTemp(t1);
    setM2DCUTemp(t2);
  }
  inline void setCCSTempLow(float temp) { m_ccstemplow = temp; }
  inline void setCCSTempHigh(float temp) { m_ccstemphigh = temp; }
  inline void setCCSTemp(float low, float high) {
    setCCSTempLow(low);
    setCCSTempHigh(high);
  }
  inline void setM1(float vdd1, float vdd2, float vinj, float vcc, 
		    float dcutemp) {
    setM1VDD1(vdd1);
    setM1VDD2(vdd2);
    setM1Vinj(vinj);
    setM1Vcc(vcc);
    setM1DCUTemp(dcutemp);
  }
  inline void setM2(float vdd1, float vdd2, float vinj, float vcc, 
		    float dcutemp) {
    setM2VDD1(vdd1);
    setM2VDD2(vdd2);
    setM2Vinj(vinj);
    setM2Vcc(vcc);
    setM2DCUTemp(dcutemp);
  }
  inline float getM1VDD1() const { return m_m1_vdd1; }
  inline float getM1VDD2() const { return m_m1_vdd2; }
  inline float getM2VDD1() const { return m_m2_vdd1; }
  inline float getM2VDD2() const { return m_m2_vdd2; }
  inline float getM1Vinj() const { return m_m1_vinj; }
  inline float getM2Vinj() const { return m_m2_vinj; }
  inline float getM1Vcc()  const { return m_m1_vcc; }
  inline float getM2Vcc()  const { return m_m2_vcc; }
  inline float getM1DCUTemp()  const { return m_m1_dcutemp; }
  inline float getM2DCUTemp()  const { return m_m2_dcutemp; }
  inline float getCCSTempLow() const { return m_ccstemplow; }
  inline float getCCSTempHigh() const { return m_ccstemphigh; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const DCUCCSDat* item, DCUIOV* iov)
    throw(std::runtime_error);

  void writeArrayDB(const std::map< EcalLogicID, DCUCCSDat>* data, DCUIOV* iov)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, DCUCCSDat >* fillVec, DCUIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_m1_vdd1;
  float m_m2_vdd1;
  float m_m1_vdd2;
  float m_m2_vdd2;
  float m_m1_vinj;
  float m_m2_vinj;
  float m_m1_vcc;
  float m_m2_vcc;
  float m_m1_dcutemp;
  float m_m2_dcutemp;
  float m_ccstemplow;
  float m_ccstemphigh;
};

#endif
