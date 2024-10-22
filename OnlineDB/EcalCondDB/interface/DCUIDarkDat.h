#ifndef DCUIDARKDAT_H
#define DCUIDARKDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/DCUTag.h"
#include "OnlineDB/EcalCondDB/interface/DCUIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class DCUIDarkDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  DCUIDarkDat();
  ~DCUIDarkDat() override;

  // User data methods
  inline std::string getTable() override { return "DCU_IDARK_DAT"; }

  inline void setAPDIDark(float i) { m_apdIDark = i; }
  inline float getAPDIDark() const { return m_apdIDark; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const DCUIDarkDat* item, DCUIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, DCUIDarkDat>* data, DCUIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, DCUIDarkDat>* fillVec, DCUIOV* iov) noexcept(false);

  // User data
  float m_apdIDark;
};

#endif
