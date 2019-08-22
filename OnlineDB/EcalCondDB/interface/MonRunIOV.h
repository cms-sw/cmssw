#ifndef MONRUNIOV_H
#define MONRUNIOV_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"

typedef int subrun_t;

class MonRunIOV : public IIOV {
public:
  friend class EcalCondDBInterface;

  MonRunIOV();
  ~MonRunIOV() override;

  void setID(int id);
  int getID() { return m_ID; };

  // Methods for user data
  void setMonRunTag(const MonRunTag& tag);
  MonRunTag getMonRunTag() const;
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
  inline bool operator==(const MonRunIOV& m) const {
    return (m_monRunTag == m.m_monRunTag && m_runIOV == m.m_runIOV && m_subRunNum == m.m_subRunNum &&
            m_subRunStart == m.m_subRunStart && m_subRunEnd == m.m_subRunEnd);
  }

  inline bool operator!=(const MonRunIOV& m) const { return !(*this == m); }

private:
  // User data for this IOV
  MonRunTag m_monRunTag;
  RunIOV m_runIOV;
  subrun_t m_subRunNum;
  Tm m_subRunStart;
  Tm m_subRunEnd;

  int writeDB() noexcept(false);
  void fetchParentIDs(int* monRunTagID, int* runIOVID) noexcept(false);

  void setByRun(MonRunTag* montag, RunIOV* runiov, subrun_t subrun) noexcept(false);
};

#endif
