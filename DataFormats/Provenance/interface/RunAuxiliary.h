#ifndef DataFormats_Provenance_RunAuxiliary_h
#define DataFormats_Provenance_RunAuxiliary_h

#include <iosfwd>
#include <set>

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

// Auxiliary run data that is persistent

namespace edm
{
  struct RunAuxiliary {
    RunAuxiliary() :
	processHistoryID_(),
	allEventsProcessHistories_(),
	id_(),
	beginTime_(),
	endTime_() {}
    RunAuxiliary(RunID const& theId, Timestamp const& theTime, Timestamp const& theEndTime) :
	processHistoryID_(),
	allEventsProcessHistories_(),
	id_(theId),
	beginTime_(theTime),
	endTime_(theEndTime) {}
    RunAuxiliary(RunNumber_t const& run, Timestamp const& theTime, Timestamp const& theEndTime) :
	processHistoryID_(),
	allEventsProcessHistories_(),
	id_(run),
	beginTime_(theTime),
	endTime_(theEndTime) {}
    ~RunAuxiliary() {}
    void write(std::ostream& os) const;
    ProcessHistoryID& processHistoryID() const {return processHistoryID_;}
    RunID const& id() const {return id_;}
    Timestamp const& beginTime() const {return beginTime_;}
    Timestamp const& endTime() const {return endTime_;}
    RunNumber_t run() const {return id_.run();}
    void setEndTime(Timestamp const& time) {
      if (endTime_ == Timestamp::invalidTimestamp()) endTime_ = time;
    }
    bool mergeAuxiliary(RunAuxiliary const& aux);

    // most recent process that put a RunProduct into this run
    // is the last on the list, this defines what "latest" is
    mutable ProcessHistoryID processHistoryID_;

    // allEventsProcessHistories_ contains all the ProcessHistoryIDs for all
    // events in this run seen so far.
    std::set<ProcessHistoryID> allEventsProcessHistories_;

    // Run ID
    RunID id_;
    // Times from DAQ
    Timestamp beginTime_;
    Timestamp endTime_;

  private:
    void mergeNewTimestampsIntoThis_(RunAuxiliary const& newAux);    
    void mergeNewProcessHistoryIntoThis_(RunAuxiliary const& newAux);    
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, const RunAuxiliary& p) {
    p.write(os);
    return os;
  }

}

#endif
