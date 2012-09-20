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
  ~RunIOV();

  // Methods for user data
  void setRunNumber(run_t run);
  run_t getRunNumber() const;
  void setRunStart(Tm start);
  Tm getRunStart() const;
  void setRunEnd(Tm end);
  Tm getRunEnd() const;
  void setRunTag(RunTag tag);
  RunTag getRunTag() const;
  void setID(int id);

  void setDBInsertionTime(Tm dbtime){m_dbtime=dbtime;}
  Tm getDBInsertionTime(){return m_dbtime;}
 

  // Methods from IUniqueDBObject
  int getID(){ return m_ID;} ;

  int fetchID() throw(std::runtime_error);
  int fetchIDByRunAndTag() throw(std::runtime_error);
  void setByID(int id) throw(std::runtime_error);

  // operators
  inline bool operator==(const RunIOV &r) const
    {
      return (m_runNum   == r.m_runNum &&
	      m_runStart == r.m_runStart &&
	      m_runEnd   == r.m_runEnd &&
	      m_runTag   == r.m_runTag);
    }

  inline bool operator!=(const RunIOV &r) const { return !(*this == r); }

 private:
  // User data for this IOV
  run_t m_runNum;
  Tm m_runStart;
  Tm m_runEnd;
  RunTag m_runTag;
  Tm m_dbtime;

  int writeDB() throw(std::runtime_error);
  int updateEndTimeDB() throw(std::runtime_error);
  int updateStartTimeDB() throw(std::runtime_error);

  void setByRun(RunTag* tag, run_t run) throw(std::runtime_error);
  void setByRun(std::string location, run_t run) throw(std::runtime_error);
  void setByRecentData(std::string dataTable, RunTag* tag, run_t run = (unsigned int)-1) throw(std::runtime_error);
  void setByRecentData(std::string dataTable, std::string location, run_t run) throw(std::runtime_error);


/*   void fetchEarliest(RunIOV* fillIOV, RunTag* tag) const throw(std::runtime_error); */
/*   void fetchLatest(RunIOV* fillIOV, RunTag* tag) const throw(std::runtime_error); */
/*   oracle::occi::Statement* prepareFetch(const std::string sql, RunTag* tag) const throw(std::runtime_error); */
/*   void fill(RunIOV* target, oracle::occi::ResultSet* rset) const throw(std::runtime_error); */

/*   // Methods for fetching by Tm */
/*   void fetchAt(RunIOV* iov, const Tm eventTm, RunTag* tag) const throw(std::runtime_error); */
/*   void fetchWithin(std::vector<RunIOV>* fillVec, const Tm beginTm, const Tm endTm, RunTag* tag) const throw(std::runtime_error); */

/*   // Methods for fetching by run_t */
/*   void fetchAt(RunIOV* fillIOV, const run_t run, RunTag* tag) const throw(std::runtime_error); */
/*   void fetchWithin(std::vector<RunIOV>* fillVec, const run_t beginRun, const run_t endRun, RunTag* tag) const throw(std::runtime_error); */
};

#endif
