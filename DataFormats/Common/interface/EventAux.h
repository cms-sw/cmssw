#ifndef Common_EventAux_h
#define Common_EventAux_h

#include <iosfwd>
#include "boost/shared_ptr.hpp"

#include "DataFormats/Common/interface/ProcessHistory.h"
#include "DataFormats/Common/interface/ProcessHistoryID.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/Timestamp.h"
#include "DataFormats/Common/interface/LuminosityBlockID.h"

// Aux2iliary event data that is persistent

namespace edm
{
  struct EventAux {
    EventAux() :
	processHistoryID_(),
	processHistoryPtr_(),
	id_(),
	time_(),
	luminosityBlockID_() {}
    //FIXME: keep temporarily for backwards compatibility
    explicit EventAux(EventID const& id) :
	processHistoryID_(),
	processHistoryPtr_(),
	id_(id),
	time_(),
	luminosityBlockID_() {}
    EventAux(EventID id, Timestamp const& time, LuminosityBlockID lb) :
	processHistoryID_(),
	processHistoryPtr_(),
	id_(id),
	time_(time),
	luminosityBlockID_(lb) {}
    ~EventAux() {}
    void init() const;
    void write(std::ostream& os) const;
    ProcessHistory& processHistory() const {init(); return *processHistoryPtr_;}
    ProcessHistoryID& processHistoryID() const {return processHistoryID_;}
    EventID const& id() const {return id_;}
    Timestamp const& time() const {return time_;}
    LuminosityBlockID const& luminosityBlockID() const {return luminosityBlockID_;}
    // most recently process that processed this event
    // is the last on the list, this defines what "latest" is
    mutable ProcessHistoryID processHistoryID_;
    // Transient
    mutable boost::shared_ptr<ProcessHistory> processHistoryPtr_;
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
