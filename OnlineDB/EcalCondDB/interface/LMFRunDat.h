#ifndef LMFRUNDAT_H
#define LMFRUNDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LMFRunDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  LMFRunDat();
  ~LMFRunDat();

  // User data methods
  inline std::string getTable() { return "LMF_RUN_DAT"; }

  inline void setNumEvents(int num) { m_numEvents = num; }
  inline int getNumEvents() const { return m_numEvents; }

  inline void setQualityFlag(int x) { m_status = x; }
  inline int getQualityFlag() const { return m_status; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const LMFRunDat* item, LMFRunIOV* iov )
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, LMFRunDat >* fillMap, LMFRunIOV* iov)
     throw(std::runtime_error);

  // User data
  int m_numEvents;
  int m_status;
};

#endif
