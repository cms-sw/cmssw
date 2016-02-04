#ifndef RUNDAT_H
#define RUNDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class RunDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  RunDat();
  ~RunDat();

  // User data methods
  inline std::string getTable() { return "RUN_DAT"; }

  inline void setNumEvents(int num) { m_numEvents = num; }
  inline int getNumEvents() const { return m_numEvents; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const RunDat* item, RunIOV* iov )
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, RunDat >* fillMap, RunIOV* iov)
     throw(std::runtime_error);

  // User data
  int m_numEvents;
};

#endif
