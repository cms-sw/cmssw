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
  ~DCULVRBTempsDat();

  // User data methods
  inline std::string getTable() { return "DCU_LVRB_TEMPS_DAT"; }

  inline void setT1(float temp) { m_t1 = temp; }
  inline float getT1() const { return m_t1; }

  inline void setT2(float temp) { m_t2 = temp; }
  inline float getT2() const { return m_t2; }

  inline void setT3(float temp) { m_t3 = temp; }
  inline float getT3() const { return m_t3; }

  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const DCULVRBTempsDat* item, DCUIOV* iov)
    throw(std::runtime_error);

  void writeArrayDB(const std::map< EcalLogicID, DCULVRBTempsDat>* data, DCUIOV* iov)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, DCULVRBTempsDat >* fillVec, DCUIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_t1;
  float m_t2;
  float m_t3;  
};

#endif
