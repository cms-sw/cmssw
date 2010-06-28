#ifndef DataFormats_Provenance_RunAuxiliary_h
#define DataFormats_Provenance_RunAuxiliary_h

#include <iosfwd>
#include <set>

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

// Auxiliary run data that is persistent

namespace edm {
  class RunAux;
  class RunAuxiliary {
  public:
    friend void conversion(RunAux const&, RunAuxiliary&);
    RunAuxiliary() :
	processHistoryID_(),
	id_(),
	beginTime_(),
	endTime_() {}
    RunAuxiliary(RunID const& theId, Timestamp const& theTime, Timestamp const& theEndTime) :
	processHistoryID_(),
	id_(theId),
	beginTime_(theTime),
	endTime_(theEndTime) {}
    RunAuxiliary(RunNumber_t const& run, Timestamp const& theTime, Timestamp const& theEndTime) :
	processHistoryID_(),
	id_(run),
	beginTime_(theTime),
	endTime_(theEndTime) {}
    ~RunAuxiliary() {}
    void write(std::ostream& os) const;
    ProcessHistoryID const& processHistoryID() const {return processHistoryID_;}
    void setProcessHistoryID(ProcessHistoryID const& phid) {processHistoryID_ = phid;}
    RunID const& id() const {return id_;}
    RunID& id() {return id_;}
    Timestamp const& beginTime() const {return beginTime_;}
    Timestamp const& endTime() const {return endTime_;}
    RunNumber_t run() const {return id_.run();}
    void setBeginTime(Timestamp const& time) {
      if (beginTime_ == Timestamp::invalidTimestamp()) beginTime_ = time;
    }
    void setEndTime(Timestamp const& time) {
      if (endTime_ == Timestamp::invalidTimestamp()) endTime_ = time;
    }
    void mergeAuxiliary(RunAuxiliary const& aux);

  private:
    // most recent process that put a RunProduct into this run
    // is the last on the list, this defines what "latest" is
    ProcessHistoryID processHistoryID_;

    // Run ID
    RunID id_;
    // Times from DAQ
    Timestamp beginTime_;
    Timestamp endTime_;

  private:
    void mergeNewTimestampsIntoThis_(RunAuxiliary const& newAux);    
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, const RunAuxiliary& p) {
    p.write(os);
    return os;
  }

}

#endif
