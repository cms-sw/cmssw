#ifndef DataFormats_Provenance_RunAux_h
#define DataFormats_Provenance_RunAux_h

#include <iosfwd>

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/RunID.h"

// Auxiliary run information that is persistent.
// Obsolete format, used for backward compatibility only.

namespace edm {
  class RunAuxiliary;
  class RunAux {
  public:
    RunAux() : processHistoryID_(), id_() {}
    ~RunAux() {}
    mutable ProcessHistoryID processHistoryID_;
    RunNumber_t id_;
  };
  void conversion(RunAux const& from, RunAuxiliary & to);
}
#endif
