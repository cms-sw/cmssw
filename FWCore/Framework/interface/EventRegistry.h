#ifndef EDM_EVENT_REGISTRY_H
#define EDM_EVENT_REGISTRY_H

/*----------------------------------------------------------------------
  
EventRegistry: A singleton to keep track of active events.
Event.

$Id: EventRegistry.h,v 1.1 2005/05/12 20:38:18 wmtan Exp $

----------------------------------------------------------------------*/

#include <map>
#include <stdexcept>
#include <sstream>
#include "FWCore/CoreFramework/interface/CoreFrameworkfwd.h"

namespace edm {
  class EventRegistry {
  public:
    typedef std::map<CollisionID, EventPrincipal const *> EventMap;
    static EventRegistry *instance() {
      static EventRegistry me;
      return &me;
    }
    void addEvent(CollisionID evtID, EventPrincipal const *evtPtr) {
      if (eventMap.find(evtID) != eventMap.end()) {
	std::ostringstream out;
        out << "EventRegistry::addEvent: An event with ID " << evtID << " is already in registry";
        throw std::runtime_error(out.str());
      }
      eventMap.insert(std::make_pair(evtID, evtPtr));
    }
    void removeEvent(CollisionID evtID) {
      eventMap.erase(evtID);
    }
    EventPrincipal const * getEvent(CollisionID evtID) const {
      EventMap::const_iterator it = eventMap.find(evtID);
      if (it == eventMap.end()) {
	std::ostringstream out;
        out << "EventRegistry::getEvent: No event with ID " << evtID << " was found in the registry";
        throw std::runtime_error(out.str());
      }
      return it->second;
    }
  private:
    EventMap eventMap;
  };
}
#endif
