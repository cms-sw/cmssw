#ifndef DCULVRBTEMPSDAT_H
#define DCULVRBTEMPSDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/DCUTag.h"
#include "OnlineDB/EcalCondDB/interface/DCUIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class DCULVRBTempsDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  DCULVRBTempsDat();
  ~DCULVRBTempsDat() override;

  // User data methods
  inline std::string getTable() override { return "DCU_LVRB_TEMPS_DAT"; }

  inline void setT1(float temp) { m_t1 = temp; }
  inline float getT1() const { return m_t1; }

  inline void setT2(float temp) { m_t2 = temp; }
  inline float getT2() const { return m_t2; }

  inline void setT3(float temp) { m_t3 = temp; }
  inline float getT3() const { return m_t3; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const DCULVRBTempsDat* item, DCUIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, DCULVRBTempsDat>* data, DCUIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, DCULVRBTempsDat>* fillVec, DCUIOV* iov) noexcept(false);

  // User data
  float m_t1;
  float m_t2;
  float m_t3;
};

#endif
