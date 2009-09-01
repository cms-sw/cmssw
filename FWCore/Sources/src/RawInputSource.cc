/*----------------------------------------------------------------------
$Id: RawInputSource.cc,v 1.25 2009/02/11 16:44:43 wdd Exp $
----------------------------------------------------------------------*/

#include "FWCore/Sources/interface/RawInputSource.h"
#include "DataFormats/Provenance/interface/Timestamp.h" 
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "FWCore/Framework/interface/Event.h"

namespace edm {
  RawInputSource::RawInputSource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
    InputSource(pset, desc),
    runNumber_(RunNumber_t()),
    newRun_(false),
    newLumi_(false),
    eventCached_(false) {
      setTimestamp(Timestamp::beginOfTime());
  }

  RawInputSource::~RawInputSource() {
  }

  boost::shared_ptr<RunAuxiliary>
  RawInputSource::readRunAuxiliary_() {
    newRun_ = false;
    return boost::shared_ptr<RunAuxiliary>(new RunAuxiliary(runNumber_, timestamp(), Timestamp::invalidTimestamp()));
  }

  boost::shared_ptr<LuminosityBlockAuxiliary>
  RawInputSource::readLuminosityBlockAuxiliary_() {
    newLumi_ = false;
    return boost::shared_ptr<LuminosityBlockAuxiliary>(new LuminosityBlockAuxiliary(
	runNumber_, luminosityBlockNumber_, timestamp(), Timestamp::invalidTimestamp()));
  }

  EventPrincipal*
  RawInputSource::readEvent_() {
    assert(eventCached_);
    eventCached_ = false;
    return eventPrincipalCache();
  }

  std::auto_ptr<Event>
  RawInputSource::makeEvent(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, Timestamp const& tstamp) {
    if(!runAuxiliary()) {
      newRun_ = newLumi_ = true;
      setRunAuxiliary(new RunAuxiliary(run, tstamp, Timestamp::invalidTimestamp()));
      readAndCacheRun();
      setRunPrematurelyRead();
    }
    if(!luminosityBlockAuxiliary()) {
      setLuminosityBlockAuxiliary(new LuminosityBlockAuxiliary(run, lumi, tstamp, Timestamp::invalidTimestamp()));
      newLumi_ = true;
      readAndCacheLumi();
      setLumiPrematurelyRead();
    }
    EventSourceSentry sentry(*this);
    std::auto_ptr<EventAuxiliary> aux(new EventAuxiliary(EventID(run, event),
      processGUID(), tstamp, lumi, true, EventAuxiliary::PhysicsTrigger));
    eventPrincipalCache()->fillEventPrincipal(aux, luminosityBlockPrincipal());
    eventCached_ = true;
    std::auto_ptr<Event> e(new Event(*eventPrincipalCache(), moduleDescription()));
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
    if(eventCached_) {
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
      runNumber_ = e->run();
      luminosityBlockNumber_ = e->luminosityBlock();
      return IsRun;
    } else if (e->luminosityBlock() != luminosityBlockNumber_) {
      luminosityBlockNumber_ = e->luminosityBlock();
      newLumi_ = true;
      return IsLumi;
    }
    return IsEvent;
  }

  EventPrincipal *
  RawInputSource::readIt(EventID const&) {
      throw edm::Exception(errors::LogicError,"RawInputSource::readEvent_(EventID const& eventID)")
        << "Random access read cannot be used for RawInputSource.\n"
        << "Contact a Framework developer.\n";
  }

  // Not yet implemented
  void
  RawInputSource::skip(int) {
      throw edm::Exception(errors::LogicError,"RawInputSource::skip(int offset)")
        << "Random access skip cannot be used for RawInputSource\n"
        << "Contact a Framework developer.\n";
  }

}
