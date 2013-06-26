#ifndef DCULVRVOLTAGESDAT_H
#define DCULVRVOLTAGESDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/DCUTag.h"
#include "OnlineDB/EcalCondDB/interface/DCUIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class DCULVRVoltagesDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  DCULVRVoltagesDat();
  ~DCULVRVoltagesDat();

  // User data methods
  inline std::string getTable() { return "DCU_LVR_VOLTAGES_DAT"; }

  inline void setVFE1_A(float v) { m_vfe1_A = v; }
  inline float getVFE1_A() const { return m_vfe1_A; }

  inline void setVFE2_A(float v) { m_vfe2_A = v; }
  inline float getVFE2_A() const { return m_vfe2_A; }

  inline void setVFE3_A(float v) { m_vfe3_A = v; }
  inline float getVFE3_A() const { return m_vfe3_A; }

  inline void setVFE4_A(float v) { m_vfe4_A = v; }
  inline float getVFE4_A() const { return m_vfe4_A; }

  inline void setVFE5_A(float v) { m_vfe5_A = v; }
  inline float getVFE5_A() const { return m_vfe5_A; }

  inline void setVCC(float v) { m_VCC = v; }
  inline float getVCC() const { return m_VCC; }
  
  inline void setVFE4_5_D(float v) { m_vfe4_5_D = v; }
  inline float getVFE4_5_D() const { return m_vfe4_5_D; }

  inline void setVFE1_2_3_D(float v) { m_vfe1_2_3_D = v; }
  inline float getVFE1_2_3_D() const { return m_vfe1_2_3_D; }

  inline void setBuffer(float v) { m_buffer = v; }
  inline float getBuffer() const { return m_buffer; }

  inline void setFenix(float v) { m_fenix = v; }
  inline float getFenix() const { return m_fenix; }

  inline void setV43_A(float v) { m_V43_A = v; }
  inline float getV43_A() const { return m_V43_A; }

  inline void setOCM(float v) { m_OCM = v; }
  inline float getOCM() const { return m_OCM; }

  inline void setGOH(float v) { m_GOH = v; }
  inline float getGOH() const { return m_GOH; }

  inline void setINH(float v) { m_INH = v; }
  inline float getINH() const { return m_INH; }

  inline void setV43_D(float v) { m_V43_D = v; }
  inline float getV43_D() const { return m_V43_D; }


 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const DCULVRVoltagesDat* item, DCUIOV* iov)
    throw(std::runtime_error);

  void writeArrayDB(const std::map< EcalLogicID, DCULVRVoltagesDat>* data, DCUIOV* iov)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, DCULVRVoltagesDat >* fillVec, DCUIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_vfe1_A;
  float m_vfe2_A;
  float m_vfe3_A;
  float m_vfe4_A;
  float m_vfe5_A;
  float m_VCC;
  float m_vfe4_5_D;
  float m_vfe1_2_3_D;
  float m_buffer;
  float m_fenix;
  float m_V43_A;
  float m_OCM;
  float m_GOH;
  float m_INH;
  float m_V43_D;
};

#endif
