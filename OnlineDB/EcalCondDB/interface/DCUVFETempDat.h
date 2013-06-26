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
  ~DCUVFETempDat();

  // User data methods
  inline std::string getTable() { return "DCU_VFE_TEMP_DAT"; }

  inline void setVFETemp(float temp) { m_vfeTemp = temp; }
  inline float getVFETemp() const { return m_vfeTemp; }
  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const DCUVFETempDat* item, DCUIOV* iov)
    throw(std::runtime_error);

  void writeArrayDB(const std::map< EcalLogicID, DCUVFETempDat>* data, DCUIOV* iov)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, DCUVFETempDat >* fillVec, DCUIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_vfeTemp;
  
};

#endif
