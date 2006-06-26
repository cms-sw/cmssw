#ifndef Common_EventAux_h
#define Common_EventAux_h

#include "DataFormats/Common/interface/ProcessNameList.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/Timestamp.h"

// Auxiliary event data that is persistent

namespace edm
{
  struct EventAux {
    EventAux() : process_history_(), id_() {}
    //FIXME: keep temporarily for backwards compatibility
    explicit EventAux(EventID const& id) : process_history_(), id_(id) {}
    EventAux(EventID id, Timestamp const& time) : process_history_(), id_(id), time_(time) {}
    ~EventAux() {}

    ProcessNameList const& processHistory() const {return process_history_;}
    EventID const& id() const {return id_;}
    Timestamp const& time() const {return time_;}

    // most recently process that processed this event
    // is the last on the list, this defines what "latest" is
    ProcessNameList process_history_;
    // Event ID
    EventID id_;
    // Time from DAQ
    Timestamp time_;
  };
}

#endif
