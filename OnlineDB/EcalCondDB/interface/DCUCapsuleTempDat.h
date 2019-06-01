#ifndef DCUCAPSULETEMPDAT_H
#define DCUCAPSULETEMPDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/DCUTag.h"
#include "OnlineDB/EcalCondDB/interface/DCUIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class DCUCapsuleTempDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  DCUCapsuleTempDat();
  ~DCUCapsuleTempDat() override;

  // User data methods
  inline std::string getTable() override { return "DCU_CAPSULE_TEMP_DAT"; }

  inline void setCapsuleTemp(float temp) { m_capsuleTemp = temp; }
  inline float getCapsuleTemp() const { return m_capsuleTemp; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const DCUCapsuleTempDat* item, DCUIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, DCUCapsuleTempDat>* data, DCUIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, DCUCapsuleTempDat>* fillVec, DCUIOV* iov) noexcept(false);

  // User data
  float m_capsuleTemp;
};

#endif
