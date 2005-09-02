#ifndef Framework_EventRegistry_h
#define Framework_EventRegistry_h

/*----------------------------------------------------------------------
  
EventRegistry: A singleton to keep track of active events.
Event.

$Id: EventRegistry.h,v 1.7 2005/09/01 05:36:25 wmtan Exp $

----------------------------------------------------------------------*/

#include <map>
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {
  class EventRegistry {
  public:
    typedef std::map<EventID, EventPrincipal const *> EventMap;
    static EventRegistry *instance();
    void addEvent(EventID const& evtID, EventPrincipal const *evtPtr);
    void removeEvent(EventID const& evtID) {
      eventMap.erase(evtID);
    }
    EventPrincipal const * getEvent(EventID const& evtID) const;

    class Operate {
    public:
      Operate(EventID const& id, EventPrincipal const* ptr):
	id_(id),reg_(EventRegistry::instance())
      { reg_->addEvent(id_,ptr); }
      ~Operate()
      { reg_->removeEvent(id_); }
    private:
      EventID id_;
      EventRegistry* reg_;
    };
  private:
    EventMap eventMap;
  };
}
#endif
