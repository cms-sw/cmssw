/*----------------------------------------------------------------------
$Id: RawInputSource.cc,v 1.16 2007/12/03 00:41:05 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Sources/interface/RawInputSource.h"
#include "DataFormats/Provenance/interface/Timestamp.h" 
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/Event.h"

namespace edm {
  RawInputSource::RawInputSource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
    InputSource(pset, desc),
    runNumber_(RunNumber_t()),
    newRun_(true),
    newLumi_(true),
    ep_(0) {
      setTimestamp(Timestamp::beginOfTime());
  }

  RawInputSource::~RawInputSource() {
  }

  void
  RawInputSource::setRun(RunNumber_t r) {
    // Do nothing if the run is not changed.
    if (r != runNumber_) {
      runNumber_ = r;
      newRun_ = newLumi_ = true;
      resetLuminosityBlockPrincipal();
      resetRunPrincipal();
    }
  }

  void
  RawInputSource::setLumi(LuminosityBlockNumber_t lb) {
    if (lb != luminosityBlockNumber_) {
      luminosityBlockNumber_ = lb;
      newLumi_ = true;
      resetLuminosityBlockPrincipal();
    }
  }

  boost::shared_ptr<RunPrincipal>
  RawInputSource::readRun_() {
    newRun_ = false;
    return boost::shared_ptr<RunPrincipal>(
	new RunPrincipal(runNumber_,
			 timestamp(),
			 Timestamp::invalidTimestamp(),
			 productRegistry(),
			 processConfiguration()));
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  RawInputSource::readLuminosityBlock_() {
    newLumi_ = false;
    return boost::shared_ptr<LuminosityBlockPrincipal>(
	new LuminosityBlockPrincipal(luminosityBlockNumber_,
				     timestamp(),
				     Timestamp::invalidTimestamp(),
				     productRegistry(),
				     runPrincipal(),
				     processConfiguration()));
  }

  std::auto_ptr<EventPrincipal>
  RawInputSource::readEvent_(boost::shared_ptr<LuminosityBlockPrincipal>) {
    assert(ep_.get() != 0);
    return ep_;
  }

  std::auto_ptr<Event>
  RawInputSource::makeEvent(EventID & eventId, Timestamp const& tstamp) {
    eventId = EventID(runNumber_, eventId.event());
    ep_ = std::auto_ptr<EventPrincipal>(
	new EventPrincipal(eventId, tstamp,
	productRegistry(), luminosityBlockPrincipal(), processConfiguration(), true, EventAuxiliary::Data));
    std::auto_ptr<Event> e(new Event(*ep_, moduleDescription()));
    return e;
  }


  InputSource::ItemType 
  RawInputSource::getNextItemType() {
    if (state() == IsInvalid) {
      return IsFile;
    }
    if (newRun_) {
      return IsRun;
    }
    if (newLumi_) {
      return IsLumi;
    }
    if(ep_.get() != 0) {
      return IsEvent;
    }
    std::auto_ptr<Event> e(readOneEvent());
    if (e.get() == 0) {
      return IsStop;
    } else {
      e->commit_();
    }
    if (e->run() != runNumber_) {
      newRun_ = newLumi_ = true;
      resetLuminosityBlockPrincipal();
      resetRunPrincipal();
      runNumber_ = e->run();
      luminosityBlockNumber_ = e->luminosityBlock();
      return IsRun;
    } else if (e->luminosityBlock() != luminosityBlockNumber_) {
      luminosityBlockNumber_ = e->luminosityBlock();
      newLumi_ = true;
      resetLuminosityBlockPrincipal();
      return IsLumi;
    }
    return IsEvent;
  }

  std::auto_ptr<EventPrincipal>
  RawInputSource::readIt(EventID const&) {
      throw cms::Exception("LogicError","RawInputSource::readEvent_(EventID const& eventID)")
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
