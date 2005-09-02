#include <stdexcept>
#include <sstream>
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/EventRegistry.h"

namespace edm {

  EventRegistry *
  EventRegistry::instance() {
    static EventRegistry me;
    return &me;
  }

  void
  EventRegistry::addEvent(EventID const& evtID, EventPrincipal const *evtPtr) {
    if (eventMap.find(evtID) != eventMap.end()) {
      throw edm::Exception(edm::errors::InsertFailure,"AlreadyPresent")
	<< "EventRegistry::addEvent: An event with ID "
	<< evtID << " is already in registry";
      }
      eventMap.insert(std::make_pair(evtID, evtPtr));
    }

  EventPrincipal const *
  EventRegistry::getEvent(EventID const& evtID) const {
    EventMap::const_iterator it = eventMap.find(evtID);
    if (it == eventMap.end()) {
      throw edm::Exception(edm::errors::NotFound,"Find")
	<< "EventRegistry::getEvent: No event with ID "
	<< evtID << " was found in the registry";
      }
      return it->second;
    }
}
