#ifndef DCUVFETEMPDAT_H
#define DCUVFETEMPDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/DCUTag.h"
#include "OnlineDB/EcalCondDB/interface/DCUIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class DCUVFETempDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  DCUVFETempDat();
  ~DCUVFETempDat() override;

  // User data methods
  inline std::string getTable() override { return "DCU_VFE_TEMP_DAT"; }

  inline void setVFETemp(float temp) { m_vfeTemp = temp; }
  inline float getVFETemp() const { return m_vfeTemp; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const DCUVFETempDat* item, DCUIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, DCUVFETempDat>* data, DCUIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, DCUVFETempDat>* fillVec, DCUIOV* iov) noexcept(false);

  // User data
  float m_vfeTemp;
};

#endif
