/*----------------------------------------------------------------------
$Id: RawInputSource.cc,v 1.15 2007/11/28 17:59:19 wmtan Exp $
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
    noMoreEvents_(false),
    newRun_(true),
    newLumi_(true),
    ep_(0),
    lbp_() {
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
    }
  }

  void
  RawInputSource::setLumi(LuminosityBlockNumber_t lb) {
    if (lb != luminosityBlockNumber_) {
      luminosityBlockNumber_ = lb;
      newLumi_ = true;
    }
  }

  boost::shared_ptr<RunPrincipal>
  RawInputSource::readRun_() {
    assert(newRun_);
    newRun_ = false;
    return boost::shared_ptr<RunPrincipal>(
	new RunPrincipal(runNumber_,
			 timestamp(),
			 Timestamp::invalidTimestamp(),
			 productRegistry(),
			 processConfiguration()));
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  RawInputSource::readLuminosityBlock_(boost::shared_ptr<RunPrincipal> rp) {
    assert(!newRun_);
    assert(newLumi_);
    newLumi_ = false;
    lbp_ = boost::shared_ptr<LuminosityBlockPrincipal>(
	new LuminosityBlockPrincipal(luminosityBlockNumber_,
				     timestamp(),
				     Timestamp::invalidTimestamp(),
				     productRegistry(),
				     rp,
				     processConfiguration()));
    readAhead();
    return lbp_;
  }

  std::auto_ptr<EventPrincipal>
  RawInputSource::readEvent_(boost::shared_ptr<LuminosityBlockPrincipal>) {
    assert(!newRun_);
    assert(!newLumi_);
    assert(ep_.get() != 0);
    std::auto_ptr<EventPrincipal> result = ep_;
    readAhead();
    return result;
  }

  std::auto_ptr<Event>
  RawInputSource::makeEvent(EventID & eventId, Timestamp const& tstamp) {
    eventId = EventID(runNumber_, eventId.event());
    ep_ = std::auto_ptr<EventPrincipal>(
	new EventPrincipal(eventId, tstamp,
	productRegistry(), lbp_, processConfiguration(), true, EventAuxiliary::Data));
    std::auto_ptr<Event> e(new Event(*ep_, moduleDescription()));
    return e;
  }

  void
  RawInputSource::readAhead() {
    assert(ep_.get() == 0);
    if (limitReached()) {
      return;
    }
    std::auto_ptr<Event> e(readOneEvent());
    if (e.get() == 0) {
      noMoreEvents_ = true;
    } else {
      e->commit_();
    }
  }

  InputSource::ItemType
  RawInputSource::getNextItemType() const {
    if (noMoreEvents_) {
      return InputSource::IsStop;
    } else if (newRun_) {
      return InputSource::IsRun;
    } else if (newLumi_) {
      return InputSource::IsLumi;
    }
    return InputSource::IsEvent;
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
