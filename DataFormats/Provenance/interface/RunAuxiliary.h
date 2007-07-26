#ifndef DataFormats_Provenance_RunAuxiliary_h
#define DataFormats_Provenance_RunAuxiliary_h

#include <iosfwd>

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

// Auxiliary run data that is persistent

namespace edm
{
  struct RunAuxiliary {
    RunAuxiliary() :
	processHistoryID_(),
	id_(),
	time_() {}
    RunAuxiliary(RunID const& theId, Timestamp const& theTime) :
	processHistoryID_(),
	id_(theId),
	time_(theTime) {}
    RunAuxiliary(RunNumber_t const& run, Timestamp const& theTime) :
	processHistoryID_(),
	id_(run),
	time_(theTime) {}
    ~RunAuxiliary() {}
    void write(std::ostream& os) const;
    ProcessHistoryID& processHistoryID() const {return processHistoryID_;}
    RunID const& id() const {return id_;}
    Timestamp const& time() const {return time_;}
    RunNumber_t run() const {return id_.run();}

    // most recent process that processed this run
    // is the last on the list, this defines what "latest" is
    mutable ProcessHistoryID processHistoryID_;
    // Run ID
    RunID id_;
    // Time from DAQ
    Timestamp time_;
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, const RunAuxiliary& p) {
    p.write(os);
    return os;
  }

}

#endif
