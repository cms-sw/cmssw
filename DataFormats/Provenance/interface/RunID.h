#ifndef DataFormats_Provenance_RunID_h
#define DataFormats_Provenance_RunID_h
// -*- C++ -*-
//
// Package:     DataFormats/Provenance
// Class  :     RunID
//
/**\class edm::RunID

 Description: Holds run number

*/

#include <iosfwd>

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

namespace edm {

  class RunID {
  public:
    RunID() : run_(invalidRunNumber) {}
    explicit RunID(RunNumber_t iRun) : run_(iRun) {}

    // ---------- const member functions ---------------------
    RunNumber_t run() const { return run_; }

    //moving from one RunID to another one
    RunID next() const { return RunID(run_ + 1); }
    RunID previous() const {
      if (run_ != 0) {
        return RunID(run_ - 1);
      }
      return RunID(0);
    }

    bool operator==(RunID const& iRHS) const { return iRHS.run_ == run_; }
    bool operator!=(RunID const& iRHS) const { return !(*this == iRHS); }

    bool operator<(RunID const& iRHS) const { return run_ < iRHS.run_; }
    bool operator<=(RunID const& iRHS) const { return run_ <= iRHS.run_; }
    bool operator>(RunID const& iRHS) const { return run_ > iRHS.run_; }
    bool operator>=(RunID const& iRHS) const { return run_ >= iRHS.run_; }
    // ---------- static functions ---------------------------

    static RunNumber_t maxRunNumber() { return 0xFFFFFFFFU; }

    static RunID firstValidRun() { return RunID(1); }

  private:
    // ---------- member data --------------------------------
    RunNumber_t run_;
  };

  std::ostream& operator<<(std::ostream& oStream, RunID const& iID);

}  // namespace edm
#endif
