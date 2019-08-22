#ifndef DCUIDARKPEDDAT_H
#define DCUIDARKPEDDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/DCUTag.h"
#include "OnlineDB/EcalCondDB/interface/DCUIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class DCUIDarkPedDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  DCUIDarkPedDat();
  ~DCUIDarkPedDat() override;

  // User data methods
  inline std::string getTable() override { return "DCU_IDARK_PED_DAT"; }

  inline void setPed(float temp) { m_ped = temp; }
  inline float getPed() const { return m_ped; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const DCUIDarkPedDat* item, DCUIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, DCUIDarkPedDat>* data, DCUIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, DCUIDarkPedDat>* fillVec, DCUIOV* iov) noexcept(false);

  // User data
  float m_ped;
};

#endif
