/*----------------------------------------------------------------------
$Id: RawInputSource.cc,v 1.4 2006/07/06 19:11:43 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/RawInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/Event.h"

namespace edm {
  RawInputSource::RawInputSource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
    InputSource(pset, desc),
    remainingEvents_(maxEvents()),
    runNumber_(RunNumber_t()),
    oldRunNumber_(RunNumber_t()),
    ep_(),
    luminosityBlockPrincipal_()
  { }

  RawInputSource::~RawInputSource() {
  }

  void
  RawInputSource::setRun(RunNumber_t r) {
    runNumber_ = r;
  }

  std::auto_ptr<EventPrincipal>
  RawInputSource::read() {
    if (runNumber_ != oldRunNumber_ || luminosityBlockPrincipal_.get() == 0) {
      oldRunNumber_ = runNumber_;
      boost::shared_ptr<RunPrincipal const> runPrincipal(new RunPrincipal(runNumber_, productRegistry()));
      luminosityBlockPrincipal_ = boost::shared_ptr<LuminosityBlockPrincipal const>(
                        new LuminosityBlockPrincipal(1UL, productRegistry(), runPrincipal));
    }
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
    ep_ = std::auto_ptr<EventPrincipal>(new EventPrincipal(eventId,
							   Timestamp(tstamp),
							   productRegistry(),
							   luminosityBlockPrincipal_));
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
