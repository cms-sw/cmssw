#ifndef MONPNREDDAT_H
#define MONPNREDDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonPNRedDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  MonPNRedDat();
  ~MonPNRedDat();

  // User data methods
  inline std::string getTable() { return "MON_PN_RED_DAT"; }

  inline void setADCMeanG1(float mean) { m_adcMeanG1 = mean; }
  inline float getADCMeanG1() const { return m_adcMeanG1; }

  inline void setADCRMSG1(float mean) { m_adcRMSG1 = mean; }
  inline float getADCRMSG1() const { return m_adcRMSG1; }

  inline void setADCMeanG16(float mean) { m_adcMeanG16 = mean; }
  inline float getADCMeanG16() const { return m_adcMeanG16; }

  inline void setADCRMSG16(float mean) { m_adcRMSG16 = mean; }
  inline float getADCRMSG16() const { return m_adcRMSG16; }

  inline void setPedMeanG1(float mean) { m_pedMeanG1 = mean; }
  inline float getPedMeanG1() const { return m_pedMeanG1; }

  inline void setPedRMSG1(float mean) { m_pedRMSG1 = mean; }
  inline float getPedRMSG1() const { return m_pedRMSG1; }

  inline void setPedMeanG16(float mean) { m_pedMeanG16 = mean; }
  inline float getPedMeanG16() const { return m_pedMeanG16; }

  inline void setPedRMSG16(float mean) { m_pedRMSG16 = mean; }
  inline float getPedRMSG16() const { return m_pedRMSG16; }

  inline void setTaskStatus(bool status) { m_taskStatus = status; }
  inline bool getTaskStatus() const { return m_taskStatus; }
  
 private:
  void prepareWrite() 
    noexcept(false);

  void writeDB(const EcalLogicID* ecid, const MonPNRedDat* item, MonRunIOV* iov)
    noexcept(false);

  void writeArrayDB(const std::map< EcalLogicID, MonPNRedDat >* data, MonRunIOV* iov)
    noexcept(false);

  void fetchData(std::map< EcalLogicID, MonPNRedDat >* fillVec, MonRunIOV* iov)
     noexcept(false);

  // User data
  float m_adcMeanG1;
  float m_adcRMSG1;
  float m_adcMeanG16;
  float m_adcRMSG16;
  float m_pedMeanG1;
  float m_pedRMSG1;
  float m_pedMeanG16;
  float m_pedRMSG16;
  bool m_taskStatus;
};

#endif
