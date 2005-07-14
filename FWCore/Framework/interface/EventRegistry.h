#ifndef EDM_EVENT_REGISTRY_H
#define EDM_EVENT_REGISTRY_H

/*----------------------------------------------------------------------
  
EventRegistry: A singleton to keep track of active events.
Event.

$Id: EventRegistry.h,v 1.2 2005/06/28 04:46:02 jbk Exp $

----------------------------------------------------------------------*/

#include <map>
#include <stdexcept>
#include <sstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/EDMException.h"

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
        throw edm::Exception(edm::errors::InsertFailure,"AlreadyPresent")
	  << "EventRegistry::addEvent: An event with ID "
	  << evtID << " is already in registry";
      }
      eventMap.insert(std::make_pair(evtID, evtPtr));
    }
    void removeEvent(CollisionID evtID) {
      eventMap.erase(evtID);
    }
    EventPrincipal const * getEvent(CollisionID evtID) const {
      EventMap::const_iterator it = eventMap.find(evtID);
      if (it == eventMap.end()) {
        throw edm::Exception(edm::errors::NotFound,"Find")
	  << "EventRegistry::getEvent: No event with ID "
	  << evtID << " was found in the registry";
      }
      return it->second;
    }
  private:
    EventMap eventMap;
  };
}
#endif
