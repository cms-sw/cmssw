#ifndef DataFormats_Provenance_RunAuxiliary_h
#define DataFormats_Provenance_RunAuxiliary_h

#include <iosfwd>

#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/RunID.h"

// Auxiliary run data that is persistent

namespace edm
{
  struct RunAuxiliary {
    RunAuxiliary() :
	processHistoryID_(),
	id_() {}
    explicit RunAuxiliary(RunID const& theId) :
	processHistoryID_(),
	id_(theId) {}
    explicit RunAuxiliary(RunNumber_t const& run) :
	processHistoryID_(),
	id_(run) {}
    ~RunAuxiliary() {}
    void write(std::ostream& os) const;
    ProcessHistoryID& processHistoryID() const {return processHistoryID_;}
    RunID const& id() const {return id_;}
    RunNumber_t run() const {return id_.run();}

    // most recent process that processed this run
    // is the last on the list, this defines what "latest" is
    mutable ProcessHistoryID processHistoryID_;
    // Run ID
    RunID id_;
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, const RunAuxiliary& p) {
    p.write(os);
    return os;
  }

}

#endif
