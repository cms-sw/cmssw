#ifndef MONPNPEDDAT_H
#define MONPNPEDDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonPNPedDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  MonPNPedDat();
  ~MonPNPedDat() override;

  // User data methods
  inline std::string getTable() override { return "MON_PN_PED_DAT"; }

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
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const MonPNPedDat* item, MonRunIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, MonPNPedDat>* data, MonRunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, MonPNPedDat>* fillVec, MonRunIOV* iov) noexcept(false);

  // User data
  float m_pedMeanG1;
  float m_pedRMSG1;
  float m_pedMeanG16;
  float m_pedRMSG16;
  bool m_taskStatus;
};

#endif
