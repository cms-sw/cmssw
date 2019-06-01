#ifndef DCUCAPSULETEMPRAWDAT_H
#define DCUCAPSULETEMPRAWDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/DCUTag.h"
#include "OnlineDB/EcalCondDB/interface/DCUIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class DCUCapsuleTempRawDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  DCUCapsuleTempRawDat();
  ~DCUCapsuleTempRawDat() override;

  // User data methods
  inline std::string getTable() override { return "DCU_CAPSULE_TEMP_RAW_DAT"; }

  inline void setCapsuleTempADC(float adc) { m_capsuleTempADC = adc; }
  inline float getCapsuleTempADC() const { return m_capsuleTempADC; }

  inline void setCapsuleTempRMS(float rms) { m_capsuleTempRMS = rms; }
  inline float getCapsuleTempRMS() const { return m_capsuleTempRMS; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const DCUCapsuleTempRawDat* item, DCUIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, DCUCapsuleTempRawDat>* data, DCUIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, DCUCapsuleTempRawDat>* fillVec, DCUIOV* iov) noexcept(false);

  // User data
  float m_capsuleTempADC;
  float m_capsuleTempRMS;
};

#endif
