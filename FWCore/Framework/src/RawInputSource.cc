/*----------------------------------------------------------------------
$Id: RawInputSource.cc,v 1.6 2006/12/19 00:28:56 wmtan Exp $
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
    luminosityBlockID_(LuminosityBlockID()),
    oldLuminosityBlockID_(LuminosityBlockID()),
    ep_(),
    luminosityBlockPrincipal_()
  { }

  RawInputSource::~RawInputSource() {
  }

  void
  RawInputSource::setRun(RunNumber_t r) {
    runNumber_ = r;
    luminosityBlockID_ = 1;
  }

  void
  RawInputSource::setLumi(LuminosityBlockID lb) {
    luminosityBlockID_ = lb;
  }

  std::auto_ptr<EventPrincipal>
  RawInputSource::read() {
    if (oldRunNumber_ != runNumber_ || luminosityBlockPrincipal_.get() == 0) {
      oldRunNumber_ = runNumber_;
      oldLuminosityBlockID_ = luminosityBlockID_;
      boost::shared_ptr<RunPrincipal const> runPrincipal(new RunPrincipal(runNumber_, productRegistry()));
      luminosityBlockPrincipal_ = boost::shared_ptr<LuminosityBlockPrincipal const>(
                        new LuminosityBlockPrincipal(luminosityBlockID_, productRegistry(), runPrincipal));
    } else if (oldLuminosityBlockID_ != luminosityBlockID_) {
      oldLuminosityBlockID_ = luminosityBlockID_;
      boost::shared_ptr<RunPrincipal const> runPrincipal = luminosityBlockPrincipal_->runPrincipalConstSharedPtr();
      luminosityBlockPrincipal_ = boost::shared_ptr<LuminosityBlockPrincipal const>(
                        new LuminosityBlockPrincipal(luminosityBlockID_, productRegistry(), runPrincipal));
    }
    if (remainingEvents_ != 0) {
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
