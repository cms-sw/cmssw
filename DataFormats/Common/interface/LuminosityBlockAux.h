#ifndef Common_LuminosityBlockAux_h
#define Common_LuminosityBlockAux_h

#include <iosfwd>
#include "boost/shared_ptr.hpp"

#include "DataFormats/Common/interface/ProcessHistory.h"
#include "DataFormats/Common/interface/ProcessHistoryID.h"
#include "DataFormats/Common/interface/LuminosityBlockID.h"
#include "DataFormats/Common/interface/RunID.h"
#include "DataFormats/Common/interface/Timestamp.h"

// Auxiliary luminosity block data that is persistent

namespace edm
{
  struct LuminosityBlockAux {
    typedef LuminosityBlockID IDValue;
    typedef Timestamp TimeValue;
    typedef LuminosityBlockID LumiValue;
    LuminosityBlockAux() :
	processHistoryID_(),
	id_(),
	runID_(),
	time_() {}
    LuminosityBlockAux(LuminosityBlockID const& theId, RunNumber_t const& theRun, Timestamp const& theTime) :
	processHistoryID_(),
	id_(theId),
	runID_(theRun),
	time_(theTime) {}
    ~LuminosityBlockAux() {}
    void write(std::ostream& os) const;
    ProcessHistoryID& processHistoryID() const {return processHistoryID_;}
    LuminosityBlockID const& id() const {return id_;}
    Timestamp const& time() const {return time_;}

    // most recent process that processed this lumi block
    // is the last on the list, this defines what "latest" is
    mutable ProcessHistoryID processHistoryID_;
    // LuminosityBlock ID
    LuminosityBlockID id_;
    // Associated run number
    RunNumber_t runID_;
    // Time from DAQ
    Timestamp time_;
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, const LuminosityBlockAux& p) {
    p.write(os);
    return os;
  }

}

#endif
