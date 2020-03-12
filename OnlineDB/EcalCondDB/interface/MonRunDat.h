#ifndef MONRUNDAT_H
#define MONRUNDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonRunOutcomeDef.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonRunDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  MonRunDat();
  ~MonRunDat() override;

  // User data methods
  inline std::string getTable() override { return "MON_RUN_DAT"; }

  inline void setNumEvents(int num) { m_numEvents = num; }
  inline int getNumEvents() const { return m_numEvents; }

  inline void setMonRunOutcomeDef(const MonRunOutcomeDef& outcomeDef) { m_outcomeDef = outcomeDef; }
  inline MonRunOutcomeDef getMonRunOutcomeDef() const { return m_outcomeDef; }

  inline void setRootfileName(std::string name) { m_rootfileName = name; }
  inline std::string getRootfileName() const { return m_rootfileName; }

  inline void setTaskList(int list) { m_taskList = list; }
  inline int getTaskList() const { return m_taskList; }

  inline void setTaskOutcome(int outcome) { m_taskOutcome = outcome; }
  inline int getTaskOutcome() const { return m_taskOutcome; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const MonRunDat* item, MonRunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, MonRunDat>* fillMap, MonRunIOV* iov) noexcept(false);

  // User data
  int m_numEvents;
  MonRunOutcomeDef m_outcomeDef;
  std::string m_rootfileName;
  int m_taskList;
  int m_taskOutcome;
};

#endif
