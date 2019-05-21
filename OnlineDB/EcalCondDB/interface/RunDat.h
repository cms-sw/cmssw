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
  ~RunDat() override;

  // User data methods
  inline std::string getTable() override { return "RUN_DAT"; }

  inline void setNumEvents(int num) { m_numEvents = num; }
  inline int getNumEvents() const { return m_numEvents; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const RunDat* item, RunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, RunDat>* fillMap, RunIOV* iov) noexcept(false);

  // User data
  int m_numEvents;
};

#endif
