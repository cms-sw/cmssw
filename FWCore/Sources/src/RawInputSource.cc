/*----------------------------------------------------------------------
$Id: RawInputSource.cc,v 1.1 2007/05/01 20:21:57 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Sources/interface/RawInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"

namespace edm {
  RawInputSource::RawInputSource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
    InputSource(pset, desc),
    remainingEvents_(maxEvents()),
    runNumber_(RunNumber_t()),
    ep_()
  { }

  RawInputSource::~RawInputSource() {
  }

  void
  RawInputSource::setRun(RunNumber_t r) {
    // Do nothing if the run is not changed.
    if (r != runNumber_) {
      runNumber_ = r;
    }
  }

  void
  RawInputSource::setLumi(LuminosityBlockNumber_t lb) {
    luminosityBlockNumber_ = lb;
  }

  std::auto_ptr<EventPrincipal>
  RawInputSource::read() {
    if (remainingEvents_ == 0) {
      return std::auto_ptr<EventPrincipal>(0); 
    }
    std::auto_ptr<Event> e(readOneEvent());
    if (e.get() == 0) {
      return std::auto_ptr<EventPrincipal>(0); 
    }
    --remainingEvents_;
    e->commit_();
    return ep_;
  }

  std::auto_ptr<Event>
  RawInputSource::makeEvent(EventID & eventId, Timestamp const& tstamp) {
    eventId = EventID(runNumber_, eventId.event());
    ep_ = std::auto_ptr<EventPrincipal>(
	new EventPrincipal(eventId, Timestamp(tstamp),
	productRegistry(), luminosityBlockNumber_, processConfiguration(), true));
    std::auto_ptr<Event> e(new Event(*ep_, moduleDescription()));
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
