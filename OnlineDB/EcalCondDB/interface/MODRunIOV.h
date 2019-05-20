#ifndef MODRUNIOV_H
#define MODRUNIOV_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"

typedef int subrun_t;

class MODRunIOV : public IIOV {
public:
  friend class EcalCondDBInterface;

  MODRunIOV();
  ~MODRunIOV() override;

  void setID(int id);
  int getID() { return m_ID; };

  // Methods for user data
  void setRunIOV(const RunIOV& iov);
  RunIOV getRunIOV();
  void setSubRunNumber(subrun_t subrun);
  run_t getSubRunNumber() const;
  void setSubRunStart(const Tm& start);
  Tm getSubRunStart() const;
  void setSubRunEnd(const Tm& end);
  Tm getSubRunEnd() const;

  // Methods from IUniqueDBObject
  int fetchID() noexcept(false) override;
  void setByID(int id) noexcept(false) override;

  // Operators
  inline bool operator==(const MODRunIOV& m) const {
    return (m_runIOV == m.m_runIOV && m_subRunNum == m.m_subRunNum && m_subRunStart == m.m_subRunStart &&
            m_subRunEnd == m.m_subRunEnd);
  }

  inline bool operator!=(const MODRunIOV& m) const { return !(*this == m); }

private:
  // User data for this IOV
  RunIOV m_runIOV;
  subrun_t m_subRunNum;
  Tm m_subRunStart;
  Tm m_subRunEnd;

  int writeDB() noexcept(false);
  void fetchParentIDs(int* runIOVID) noexcept(false);

  void setByRun(RunIOV* runiov, subrun_t subrun) noexcept(false);
};

#endif
