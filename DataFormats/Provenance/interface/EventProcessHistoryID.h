#ifndef DataFormats_Provenance_EventProcessHistoryID_h
#define DataFormats_Provenance_EventProcessHistoryID_h

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

namespace edm
{
  struct EventProcessHistoryID {
    EventProcessHistoryID() : eventID_(), processHistoryID_() {}
    EventProcessHistoryID(EventID const& id, ProcessHistoryID const& ph) : eventID_(id), processHistoryID_(ph) {}
    EventID eventID_;
    ProcessHistoryID processHistoryID_;
  };
  inline
  bool operator<(EventProcessHistoryID const& lh, EventProcessHistoryID const& rh) {
      return lh.eventID_ < rh.eventID_;
  }
}


#endif
