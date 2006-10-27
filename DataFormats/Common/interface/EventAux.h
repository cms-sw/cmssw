#ifndef Common_EventAux_h
#define Common_EventAux_h

#include <iosfwd>
#include "boost/shared_ptr.hpp"

#include "DataFormats/Common/interface/ProcessHistory.h"
#include "DataFormats/Common/interface/ProcessHistoryID.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/Timestamp.h"
#include "DataFormats/Common/interface/LuminosityBlockID.h"

// Auxiliary event data that is persistent

namespace edm
{
  struct EventAux {
    typedef EventID IDValue;
    typedef Timestamp TimeValue;
    typedef LuminosityBlockID LumiValue;
    EventAux() :
	processHistoryID_(),
	id_(),
	time_(),
	luminosityBlockID_() {}
    //FIXME: keep temporarily for backwards compatibility
    explicit EventAux(EventID const& theId) :
	processHistoryID_(),
	id_(theId),
	time_(),
	luminosityBlockID_() {}
    EventAux(EventID const& theId, Timestamp const& theTime, LuminosityBlockID lb = 1UL) :
	processHistoryID_(),
	id_(theId),
	time_(theTime),
	luminosityBlockID_(lb) {}
    ~EventAux() {}
    void write(std::ostream& os) const;
    ProcessHistoryID& processHistoryID() const {return processHistoryID_;}
    EventID const& id() const {return id_;}
    Timestamp const& time() const {return time_;}
    LuminosityBlockID const& luminosityBlockID() const {return luminosityBlockID_;}
    // most recently process that processed this event
    // is the last on the list, this defines what "latest" is
    mutable ProcessHistoryID processHistoryID_;
    // Event ID
    EventID id_;
    // Time from DAQ
    Timestamp time_;
    // Associated Luminosity Block identifier.
    LuminosityBlockID luminosityBlockID_;
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, const EventAux& p) {
    p.write(os);
    return os;
  }

}

#endif
