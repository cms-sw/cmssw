#include <stdexcept>
#include <sstream>
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/EventRegistry.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "boost/thread/tss.hpp"

namespace edm {

  EventRegistry *
  EventRegistry::instance() {
    static boost::thread_specific_ptr<EventRegistry> s_registry;
    if(0 == s_registry.get()){
      s_registry.reset(new EventRegistry);
    }
    return s_registry.get();
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
  EDProduct const* 
  EventRegistry::get(EventID const& evtID, ProductID const& prodID) const {
     EventPrincipal const* ep = getEvent(evtID);
     return ep->get(prodID).wrapper();
  }
}
