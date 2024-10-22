#ifndef MONTESTPULSEDAT_H
#define MONTESTPULSEDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonTestPulseDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  MonTestPulseDat();
  ~MonTestPulseDat() override;

  // User data methods
  inline std::string getTable() override { return "MON_TEST_PULSE_DAT"; }

  inline void setADCMeanG1(float mean) { m_adcMeanG1 = mean; }
  inline float getADCMeanG1() const { return m_adcMeanG1; }

  inline void setADCRMSG1(float rms) { m_adcRMSG1 = rms; }
  inline float getADCRMSG1() const { return m_adcRMSG1; }

  inline void setADCMeanG6(float mean) { m_adcMeanG6 = mean; }
  inline float getADCMeanG6() const { return m_adcMeanG6; }

  inline void setADCRMSG6(float rms) { m_adcRMSG6 = rms; }
  inline float getADCRMSG6() const { return m_adcRMSG6; }

  inline void setADCMeanG12(float mean) { m_adcMeanG12 = mean; }
  inline float getADCMeanG12() const { return m_adcMeanG12; }

  inline void setADCRMSG12(float rms) { m_adcRMSG12 = rms; }
  inline float getADCRMSG12() const { return m_adcRMSG12; }

  inline void setTaskStatus(bool status) { m_taskStatus = status; }
  inline bool getTaskStatus() const { return m_taskStatus; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const MonTestPulseDat* item, MonRunIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, MonTestPulseDat>* data, MonRunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, MonTestPulseDat>* fillMap, MonRunIOV* iov) noexcept(false);

  // User data
  float m_adcMeanG1;
  float m_adcRMSG1;
  float m_adcMeanG6;
  float m_adcRMSG6;
  float m_adcMeanG12;
  float m_adcRMSG12;
  bool m_taskStatus;
};

#endif
