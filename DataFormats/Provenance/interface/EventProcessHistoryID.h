#ifndef DataFormats_Provenance_EventProcessHistoryID_h
#define DataFormats_Provenance_EventProcessHistoryID_h

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

namespace edm {
  class EventProcessHistoryID {
  public:
    EventProcessHistoryID() : eventID_(), processHistoryID_() {}
    EventProcessHistoryID(EventID const& id, ProcessHistoryID const& ph) : eventID_(id), processHistoryID_(ph) {}
    EventID const& eventID() const {return eventID_;}
    ProcessHistoryID const& processHistoryID() const {return processHistoryID_;}
  private:
    EventID eventID_;
    ProcessHistoryID processHistoryID_;
  };
  inline
  bool operator<(EventProcessHistoryID const& lh, EventProcessHistoryID const& rh) {
      return lh.eventID() < rh.eventID();
  }
}


#endif
