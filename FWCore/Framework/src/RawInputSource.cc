/*----------------------------------------------------------------------
$Id: RawInputSource.cc,v 1.2 2006/04/04 22:15:22 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/RawInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"

namespace edm {
  RawInputSource::RawInputSource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
    InputSource(pset, desc),
    remainingEvents_(maxEvents()),
    runNumber_(RunNumber_t()),
    ep_(0)
  { }

  RawInputSource::~RawInputSource() {
  }

  std::auto_ptr<EventPrincipal>
  RawInputSource::read() {
    if(remainingEvents_ != 0) {
      std::auto_ptr<Event> e(readOneEvent());
      if(e.get() != 0) {
        --remainingEvents_;
        e->commit_();
      }
    }
    return ep_;
  }

  std::auto_ptr<Event>
  RawInputSource::makeEvent(EventID & eventId, Timestamp const& tstamp) {
    eventId = EventID(runNumber_, eventId.event());
    ep_ = std::auto_ptr<EventPrincipal>(new EventPrincipal(eventId, Timestamp(tstamp), productRegistry()));
    std::auto_ptr<Event> e(new Event(*ep_, module()));
    return e;
  }

  std::auto_ptr<EventPrincipal>
  RawInputSource::readIt(EventID const&) {
      throw cms::Exception("LogicError","RawInputSource::read(EventID const& eventID)")
        << "Random access read cannot be used for RawInputSource.\n"
        << "Contact a Framework developer.\n";
  }

  // Not yet implemented
  void
  RawInputSource::skip(int) {
      throw cms::Exception("LogicError","RawInputSource::skip(int offset)")
        << "Random access skip cannot be used for RawInputSource\n"
        << "Contact a Framework developer.\n";
  }

}
