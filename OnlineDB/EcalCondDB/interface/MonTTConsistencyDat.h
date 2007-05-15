#ifndef MONTTCONSISTENCYDAT_H
#define MONTTCONSISTENCYDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonTTConsistencyDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  MonTTConsistencyDat();
  ~MonTTConsistencyDat();

  // User data methods
  inline std::string getTable() { return "MON_TT_CONSISTENCY_DAT"; }

  inline void setProcessedEvents(int proc) { m_processedEvents = proc; }
  inline int getProcessedEvents() const { return m_processedEvents; }

  inline void setProblematicEvents(int prob) { m_problematicEvents = prob; }
  inline int getProblematicEvents() const { return m_problematicEvents; }

  inline void setProblemsID(int id) { m_problemsID = id; }
  inline int getProblemsID() const { return m_problemsID; }

  inline void setProblemsSize(int size) { m_problemsSize = size; }
  inline int getProblemsSize() const { return m_problemsSize; }

  inline void setProblemsLV1(int LV1) { m_problemsLV1 = LV1; }
  inline int getProblemsLV1() const { return m_problemsLV1; }

  inline void setProblemsBunchX(int bunchX) { m_problemsBunchX = bunchX; }
  inline int getProblemsBunchX() const { return m_problemsBunchX; }

  inline void setTaskStatus(bool status) { m_taskStatus = status; }
  inline bool getTaskStatus() const { return m_taskStatus; }
  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const MonTTConsistencyDat* item, MonRunIOV* iov)
    throw(std::runtime_error);

  void writeArrayDB(const std::map< EcalLogicID, MonTTConsistencyDat >* data, MonRunIOV* iov)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, MonTTConsistencyDat >* fillVec, MonRunIOV* iov)
     throw(std::runtime_error);

  // User data
  int m_processedEvents;
  int m_problematicEvents;
  int m_problemsID;
  int m_problemsSize;
  int m_problemsLV1;
  int m_problemsBunchX;
  bool m_taskStatus;
  
};

#endif
