#ifndef MONCRYSTALCONSISTENCYDAT_H
#define MONCRYSTALCONSISTENCYDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonCrystalConsistencyDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  MonCrystalConsistencyDat();
  ~MonCrystalConsistencyDat();

  // User data methods
  inline std::string getTable() { return "MON_CRYSTAL_CONSISTENCY_DAT"; }

  inline void setProcessedEvents(int proc) { m_processedEvents = proc; }
  inline int getProcessedEvents() const { return m_processedEvents; }

  inline void setProblematicEvents(int prob) { m_problematicEvents = prob; }
  inline int getProblematicEvents() const { return m_problematicEvents; }

  inline void setProblemsID(int id) { m_problemsID = id; }
  inline int getProblemsID() const { return m_problemsID; }

  inline void setProblemsGainZero(int prob) { m_problemsGainZero = prob; }
  inline int getProblemsGainZero() const { return m_problemsGainZero; }

  inline void setProblemsGainSwitch(int prob) { m_problemsGainSwitch = prob; }
  inline int getProblemsGainSwitch() const { return m_problemsGainSwitch; }

  inline void setTaskStatus(bool status) { m_taskStatus = status; }
  inline bool getTaskStatus() const { return m_taskStatus; }
  
 private:
  void prepareWrite() 
    noexcept(false);

  void writeDB(const EcalLogicID* ecid, const MonCrystalConsistencyDat* item, MonRunIOV* iov)
    noexcept(false);

  void fetchData(std::map< EcalLogicID, MonCrystalConsistencyDat >* fillVec, MonRunIOV* iov)
     noexcept(false);

  void writeArrayDB(const std::map< EcalLogicID, MonCrystalConsistencyDat >* data, MonRunIOV* iov)
    noexcept(false);


  // User data
  int m_processedEvents;
  int m_problematicEvents;
  int m_problemsID;
  int m_problemsGainZero;
  int m_problemsGainSwitch;
  bool m_taskStatus;
  
};

#endif
