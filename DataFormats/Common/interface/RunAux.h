#ifndef Common_RunAux_h
#define Common_RunAux_h

#include <iosfwd>

#include "DataFormats/Common/interface/ProcessHistory.h"
#include "DataFormats/Common/interface/ProcessHistoryID.h"
#include "DataFormats/Common/interface/RunID.h"

// Auxiliary run data that is persistent

namespace edm
{
  struct RunAux {
    RunAux() :
	processHistoryID_(),
	id_() {}
    explicit RunAux(RunNumber_t const& theId) :
	processHistoryID_(),
	id_(theId) {}
    ~RunAux() {}
    void write(std::ostream& os) const;
    ProcessHistoryID& processHistoryID() const {return processHistoryID_;}
    RunNumber_t const& id() const {return id_;}

    // most recent process that processed this run
    // is the last on the list, this defines what "latest" is
    mutable ProcessHistoryID processHistoryID_;
    // Run ID
    RunNumber_t id_;
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, const RunAux& p) {
    p.write(os);
    return os;
  }

}

#endif
