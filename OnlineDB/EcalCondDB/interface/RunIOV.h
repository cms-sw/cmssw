#ifndef RUNIOV_H
#define RUNIOV_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"

typedef int run_t;

class RunIOV : public IIOV {
public:
  friend class EcalCondDBInterface;

  RunIOV();
  ~RunIOV() override;

  // Methods for user data
  void setRunNumber(run_t run);
  run_t getRunNumber() const;
  void setRunStart(const Tm& start);
  Tm getRunStart() const;
  void setRunEnd(const Tm& end);
  Tm getRunEnd() const;
  void setRunTag(const RunTag& tag);
  RunTag getRunTag() const;
  void setID(int id);

  void setDBInsertionTime(const Tm& dbtime) { m_dbtime = dbtime; }
  Tm getDBInsertionTime() { return m_dbtime; }

  // Methods from IUniqueDBObject
  int getID() { return m_ID; };

  int fetchID() noexcept(false) override;
  int fetchIDByRunAndTag() noexcept(false);
  void setByID(int id) noexcept(false) override;

  // operators
  inline bool operator==(const RunIOV& r) const {
    return (m_runNum == r.m_runNum && m_runStart == r.m_runStart && m_runEnd == r.m_runEnd && m_runTag == r.m_runTag);
  }

  inline bool operator!=(const RunIOV& r) const { return !(*this == r); }

private:
  // User data for this IOV
  run_t m_runNum;
  Tm m_runStart;
  Tm m_runEnd;
  RunTag m_runTag;
  Tm m_dbtime;

  int writeDB() noexcept(false);
  int updateEndTimeDB() noexcept(false);
  int updateStartTimeDB() noexcept(false);

  void setByRun(RunTag* tag, run_t run) noexcept(false);
  void setByRun(std::string location, run_t run) noexcept(false);
  void setByTime(std::string location, const Tm& t) noexcept(false);
  void setByRecentData(std::string dataTable, RunTag* tag, run_t run = (unsigned int)-1) noexcept(false);
  void setByRecentData(std::string dataTable, std::string location, run_t run) noexcept(false);

  /*   void fetchEarliest(RunIOV* fillIOV, RunTag* tag) const noexcept(false); */
  /*   void fetchLatest(RunIOV* fillIOV, RunTag* tag) const noexcept(false); */
  /*   oracle::occi::Statement* prepareFetch(const std::string sql, RunTag* tag) const noexcept(false); */
  /*   void fill(RunIOV* target, oracle::occi::ResultSet* rset) const noexcept(false); */

  /*   // Methods for fetching by Tm */
  /*   void fetchAt(RunIOV* iov, const Tm eventTm, RunTag* tag) const noexcept(false); */
  /*   void fetchWithin(std::vector<RunIOV>* fillVec, const Tm beginTm, const Tm endTm, RunTag* tag) const noexcept(false); */

  /*   // Methods for fetching by run_t */
  /*   void fetchAt(RunIOV* fillIOV, const run_t run, RunTag* tag) const noexcept(false); */
  /*   void fetchWithin(std::vector<RunIOV>* fillVec, const run_t beginRun, const run_t endRun, RunTag* tag) const noexcept(false); */
};

#endif
