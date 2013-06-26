#ifndef MONPEDESTALSDAT_H
#define MONPEDESTALSDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonPedestalsDat : public IDataItem {
 public:
  friend class EcalCondDBInterface; // XXX temp should not need
  MonPedestalsDat();
  ~MonPedestalsDat();

  // User data methods
  inline std::string getTable() { return "MON_PEDESTALS_DAT"; }

  inline void setPedMeanG1(float mean) { m_pedMeanG1 = mean; }
  inline float getPedMeanG1() const { return m_pedMeanG1; }

  inline void setPedMeanG6(float mean) { m_pedMeanG6 = mean; }
  inline float getPedMeanG6() const { return m_pedMeanG6; }

  inline void setPedMeanG12(float mean) { m_pedMeanG12 = mean; }
  inline float getPedMeanG12() const { return m_pedMeanG12; }

  inline void setPedRMSG1(float rms) { m_pedRMSG1 = rms; }
  inline float getPedRMSG1() const { return m_pedRMSG1; }

  inline void setPedRMSG6(float rms) { m_pedRMSG6 = rms; }
  inline float getPedRMSG6() const { return m_pedRMSG6; }

  inline void setPedRMSG12(float rms) { m_pedRMSG12 = rms; }
  inline float getPedRMSG12() const { return m_pedRMSG12; }

  inline void setTaskStatus(bool status) { m_taskStatus = status; }
  inline bool getTaskStatus() const { return m_taskStatus; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const MonPedestalsDat* item, MonRunIOV* iov)
    throw(std::runtime_error);


  void writeArrayDB(const std::map< EcalLogicID, MonPedestalsDat>* data, MonRunIOV* iov)
  throw(std::runtime_error);


  void fetchData(std::map< EcalLogicID, MonPedestalsDat >* fillMap, MonRunIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_pedMeanG1;
  float m_pedMeanG6;
  float m_pedMeanG12;
  float m_pedRMSG1;
  float m_pedRMSG6;
  float m_pedRMSG12;
  bool m_taskStatus;
};

#endif
