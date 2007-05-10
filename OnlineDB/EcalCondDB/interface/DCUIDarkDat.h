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
  ~DCUIDarkDat();

  // User data methods
  inline std::string getTable() { return "DCU_IDARK_DAT"; }

  inline void setAPDIDark(float i) { m_apdIDark = i; }
  inline float getAPDIDark() const { return m_apdIDark; }
  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const DCUIDarkDat* item, DCUIOV* iov)
    throw(std::runtime_error);

  void writeArrayDB(const std::map< EcalLogicID, DCUIDarkDat>* data, DCUIOV* iov)
    throw(std::runtime_error);


  void fetchData(std::map< EcalLogicID, DCUIDarkDat >* fillVec, DCUIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_apdIDark;
  
};

#endif
