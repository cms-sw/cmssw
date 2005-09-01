#ifndef Framework_EventRegistry_h
#define Framework_EventRegistry_h

/*----------------------------------------------------------------------
  
EventRegistry: A singleton to keep track of active events.
Event.

$Id: EventRegistry.h,v 1.6 2005/08/10 02:28:38 chrjones Exp $

----------------------------------------------------------------------*/

#include <map>
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {
  class EventRegistry {
  public:
    typedef std::map<EventID, EventPrincipal const *> EventMap;
    static EventRegistry *instance();
    void addEvent(EventID evtID, EventPrincipal const *evtPtr);
    void removeEvent(EventID evtID) {
      eventMap.erase(evtID);
    }
    EventPrincipal const * getEvent(EventID evtID) const;

    class Operate {
    public:
      Operate(EventID id, EventPrincipal const* ptr):
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
